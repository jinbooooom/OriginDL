#include "origin/pnnx/pnnx_graph.h"
#include "origin/core/config.h"
#include "origin/pnnx/operator_mapper.h"
#include "origin/pnnx/pnnx_parser.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"
#include "origin/utils/log.h"
#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#    include "origin/cuda/cuda.h"
#endif
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <vector>

namespace origin
{
namespace pnnx
{

// 构造函数仅保存路径，不读文件。实际解析在 build() -> init() 中通过 PNNXParser::parse 完成。
PNNXGraph::PNNXGraph(const std::string &param_path, const std::string &bin_path)
    : param_path_(param_path), bin_path_(bin_path)
{}

// init：解析 .param（图结构、形状、属性元信息）和 .bin（权重），得到节点列表并建立 name->node 映射。
// yolov5_infer 里从 param 解析输入尺寸（如 640x640）的格式与 PNNXParser 一致，见 get_input_shape_from_param_file。
bool PNNXGraph::init()
{
    if (unlikely(param_path_.empty() || bin_path_.empty()))
    {
        THROW_RUNTIME_ERROR("Param path or bin path is empty");
        return false;
    }

    // 解析模型文件：param 逐行解析算子（含 pnnx.Input/pnnx.Output），bin 按节点 attributes 顺序加载权重
    nodes_ = PNNXParser::parse(param_path_, bin_path_);

    if (unlikely(nodes_.empty()))
    {
        THROW_RUNTIME_ERROR("Failed to parse PNNX model or model is empty");
        return false;
    }

    // 建立节点映射，供 set_inputs/get_outputs 按名查找 "pnnx_input_0"、"pnnx_output_0"
    for (auto &node : nodes_)
    {
        node_map_[node->name] = node;
    }

    graph_state_ = GraphState::NeedBuild;
    return true;
}

// build：完整构建计算图。顺序为 init -> topological_sort -> 为每节点创建 Operator。
// topological_sort 会建立节点间的连接关系（通过构建邻接表）并确定执行顺序。
// 调用后 graph_state_ 为 Complete，方可 set_inputs/forward/get_outputs（yolov5_infer 中在 build 后即设置输入并 forward）。
void PNNXGraph::build()
{
    if (graph_state_ == GraphState::Complete)
    {
        return;  // 已经构建完成
    }

    if (graph_state_ == GraphState::NeedInit)
    {
        bool success = init();  // 解析 param/bin，得到 nodes_ 与 node_map_
        if (unlikely(!success))
        {
            THROW_RUNTIME_ERROR("Failed to initialize graph");
        }
    }

    // 拓扑排序：建立节点间连接关系（通过 output_names 与 input_names 匹配构建邻接表），
    // 计算入度并确定执行顺序，为每个非 Input/Output 节点设置 execution_order
    topological_sort();

    // 为每个计算节点创建 origin Operator（Conv2d、SiLU、Linear、YOLO Detect 等），见 operator_mapper.cpp
    // 注意：pnnx.Input 和 pnnx.Output 是特殊节点，不创建 Operator（op 为空）
    for (auto &node : nodes_)
    {
        if (node->type != "pnnx.Input" && node->type != "pnnx.Output")
        {
            node->op = OperatorMapper::create_operator(node);
            if (unlikely(!node->op))
            {
                THROW_RUNTIME_ERROR("Failed to create operator for node: {} type: {}", node->name, node->type);
            }
        }
    }

    graph_state_ = GraphState::Complete;
}

// 拓扑排序：通过 output_names 与 input_names 匹配建立节点间连接关系（构建邻接表），
// 按图依赖计算入度，BFS 得到执行顺序，为每节点赋 execution_order；forward 时按此顺序执行各 node->op
void PNNXGraph::topological_sort()
{
    std::map<std::string, int> in_degree;                   // 每个节点的入度
    std::map<std::string, std::vector<std::string>> graph;  // 邻接表

    // 初始化入度
    for (auto &node : nodes_)
    {
        in_degree[node->name] = 0;
    }

    // 构建图并计算入度
    for (auto &node : nodes_)
    {
        for (const auto &output_name : node->output_names)
        {
            // 找到使用这个输出的节点
            for (auto &other_node : nodes_)
            {
                if (other_node == node)
                    continue;

                for (const auto &input_name : other_node->input_names)
                {
                    if (input_name == output_name)
                    {
                        graph[node->name].push_back(other_node->name);
                        in_degree[other_node->name]++;
                    }
                }
            }
        }
    }

    // 拓扑排序
    std::queue<std::string> q;
    for (auto &node : nodes_)
    {
        // 跳过输入输出节点
        if (node->type == "pnnx.Input" || node->type == "pnnx.Output")
        {
            continue;
        }

        if (in_degree[node->name] == 0)
        {
            q.push(node->name);
        }
    }

    int order = 0;
    std::vector<std::shared_ptr<PNNXNode>> sorted_nodes;

    while (!q.empty())
    {
        std::string current_name = q.front();
        q.pop();

        auto node = node_map_[current_name];

        // 跳过输入输出节点
        if (node->type == "pnnx.Input" || node->type == "pnnx.Output")
        {
            continue;
        }

        node->execution_order = order++;
        sorted_nodes.push_back(node);

        // 更新下游节点的入度
        for (const auto &next_name : graph[current_name])
        {
            in_degree[next_name]--;
            if (in_degree[next_name] == 0)
            {
                q.push(next_name);
            }
        }
    }

    // 对于没有被排序的节点（可能是孤立的节点），给一个默认的 execution_order
    for (auto &node : nodes_)
    {
        if (node->type != "pnnx.Input" && node->type != "pnnx.Output" && node->execution_order < 0)
        {
            node->execution_order = order++;
            sorted_nodes.push_back(node);
        }
    }

    // 不更新 nodes_，保持原始顺序，只在执行时使用 sorted_nodes
}

// 用户调用 set_inputs("pnnx_input_0", {input}) 时进入。将 inputs 写入输入节点的 output_tensors，并传播到所有下游的 input_tensors。
// yolov5_infer 中 input 为预处理后的 batch tensor，形状如 [N,C,H,W]。
// 注意：输入节点（pnnx.Input）是特殊节点：input_count=0（input_names 为空）
// 不参与拓扑排序和执行，其 output_tensors 由用户通过本函数直接设置。
void PNNXGraph::set_inputs(const std::string &input_name, const std::vector<Tensor> &inputs)
{
    if (unlikely(graph_state_ != GraphState::Complete))
    {
        THROW_RUNTIME_ERROR("Graph must be built before setting inputs");
    }

    auto it = node_map_.find(input_name);
    if (unlikely(it == node_map_.end()))
    {
        THROW_RUNTIME_ERROR("Input node not found: {}", input_name);
    }

    auto input_node = it->second;

    // 输入节点在图中的"输出"即用户提供的 Tensor；写入 output_tensors 后通过 propagate 填到下游 input_tensors
    // 输入节点没有 input_names（input_count=0），只有 output_names（output_count>=1）
    if (input_node->output_names.size() == inputs.size())
    {
        input_node->output_tensors = inputs;

        propagate_outputs(input_node);
    }
    else
    {
        THROW_RUNTIME_ERROR("Input count mismatch: input node has {} outputs, but {} inputs provided",
                            input_node->output_names.size(), inputs.size());
    }

    // 将输入 Tensor 传递给下游节点（通过 propagate_outputs 已经完成）
    // 但为了兼容性，我们也直接设置下游节点的 input_tensors
    for (auto &node : nodes_)
    {
        for (size_t i = 0; i < node->input_names.size(); ++i)
        {
            if (node->input_names[i] == input_name)
            {
                // 设置输入 Tensor
                if (i < inputs.size())
                {
                    node->input_tensors[input_name] = inputs[i];
                }
            }
        }
    }
}

// forward：按 execution_order 依次执行各节点。每节点：收集 input_tensors（从 input_tensors 或上游 output），
// 对 Conv2d/Linear 从 node->attributes 注入 weight/bias，执行 (*node->op)(input_tensors)，再 propagate_outputs。
// yolov5_infer 在 set_inputs 后调用本函数，最后用 get_outputs("pnnx_output_0") 取检测头输出。
void PNNXGraph::forward()
{
    if (graph_state_ != GraphState::Complete)
    {
        THROW_RUNTIME_ERROR("Graph must be built before forward");
    }

    auto total_start_time = std::chrono::high_resolution_clock::now();

    // 收集所有需执行的节点（排除 pnnx.Input / pnnx.Output），按 execution_order 排序
    std::vector<std::shared_ptr<PNNXNode>> sorted_nodes;
    for (auto &node : nodes_)
    {
        if (node->type != "pnnx.Input" && node->type != "pnnx.Output")
        {
            // 检查是否有 execution_order（如果没有，说明拓扑排序时没有处理）
            if (node->execution_order < 0)
            {
                // 如果没有 execution_order，给一个默认值（放在最后执行）
                node->execution_order = 10000 + sorted_nodes.size();
            }
            sorted_nodes.push_back(node);
        }
    }

    // 按 execution_order 排序
    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
              [](const std::shared_ptr<PNNXNode> &a, const std::shared_ptr<PNNXNode> &b) {
                  return a->execution_order < b->execution_order;
              });

    // 执行排序后的节点
    size_t total_nodes    = sorted_nodes.size();
    size_t executed_count = 0;
    std::set<std::string> executed_nodes;  // 用于检测重复执行

    for (auto &node : sorted_nodes)
    {
        // 检查是否重复执行
        if (executed_nodes.find(node->name) != executed_nodes.end())
        {
            loge("ERROR: Node {} is being executed multiple times!", node->name);
            THROW_RUNTIME_ERROR("Node {} is being executed multiple times", node->name);
        }
        executed_nodes.insert(node->name);

        executed_count++;
        logd("Progress: {}/{} nodes executed (current: {})", executed_count, total_nodes, node->name);

        if (!node->op)
        {
            THROW_RUNTIME_ERROR("Operator not created for node: {}", node->name);
        }

        // 收集本节点输入：先查 input_tensors（由 set_inputs 或上游 propagate_outputs 填入），否则从上游 output 查
        std::vector<Tensor> input_tensors;
        logd("Collecting inputs for node {} (type: {})", node->name, node->type);
        for (const auto &input_name : node->input_names)
        {
            auto it = node->input_tensors.find(input_name);
            if (it != node->input_tensors.end())
            {
                input_tensors.push_back(it->second);
            }
            else
            {
                // 从上游节点获取输出（output_name_map_ 需在 build 后调用 build_output_name_map 填充才生效，否则下面会查不到而依赖后续遍历）
                auto output_it = output_name_map_.find(input_name);
                if (output_it != output_name_map_.end())
                {
                    auto &prev_node   = output_it->second.first;
                    size_t output_idx = output_it->second.second;

                    if (output_idx < prev_node->output_tensors.size())
                    {
                        input_tensors.push_back(prev_node->output_tensors[output_idx]);
                        node->input_tensors[input_name] = prev_node->output_tensors[output_idx];
                    }
                }
            }
        }

        // 带权重的算子：从 node->attributes 取 weight/bias（由 PNNXParser::load_weights 从 .bin 加载），转为 Tensor 并拼到 input_tensors
        if (node->type == "nn.Conv2d" || node->type == "nn.Linear")
        {
            // Conv2d/Linear 的 op 接口需要 (x, weight, [bias])，这里把 attributes 里的权重注入

            // 从 attributes 中获取权重（parser 已解析 shape，load_weights 已填 data）
            if (node->attributes.find("weight") != node->attributes.end())
            {
                auto &weight_attr = node->attributes["weight"];
                if (!weight_attr.data.empty() && !weight_attr.shape.empty())
                {
                    // 将权重数据转换为 Tensor
                    std::vector<size_t> weight_shape_vec;
                    for (int dim : weight_attr.shape)
                    {
                        weight_shape_vec.push_back(static_cast<size_t>(dim));
                    }
                    Shape weight_shape(weight_shape_vec);

                    // 确定设备类型（从输入 tensor 获取）
                    Device device(DeviceType::kCPU);
                    if (input_tensors.size() > 0)
                    {
                        // 从输入 tensor 获取设备（通过 to(device) 方法可以获取当前设备）
                        // 或者直接使用输入 tensor 的设备
                        device = input_tensors[0].device();
                    }
#ifdef WITH_CUDA
                    else if (cuda::is_available())
                    {
                        device = Device(DeviceType::kCUDA, 0);
                    }
#endif

                    Tensor weight_tensor(weight_attr.data, weight_shape, dtype(DataType::kFloat32).device(device));

                    // 在 x 之后插入 weight
                    if (input_tensors.size() > 0)
                    {
                        // 保存 x
                        Tensor x = input_tensors[0];
                        input_tensors.clear();
                        input_tensors.push_back(x);
                        input_tensors.push_back(weight_tensor);

                        // 如果有偏置，也添加
                        if (node->attributes.find("bias") != node->attributes.end())
                        {
                            auto &bias_attr = node->attributes["bias"];
                            if (!bias_attr.data.empty() && !bias_attr.shape.empty())
                            {
                                std::vector<size_t> bias_shape_vec;
                                for (int dim : bias_attr.shape)
                                {
                                    bias_shape_vec.push_back(static_cast<size_t>(dim));
                                }
                                Shape bias_shape(bias_shape_vec);
                                Tensor bias_tensor(bias_attr.data, bias_shape,
                                                   dtype(DataType::kFloat32).device(device));
                                input_tensors.push_back(bias_tensor);
                            }
                        }
                    }
                }
            }
        }

        // 对于 Linear 算子，需要特殊处理权重加载
        if (node->type == "nn.Linear")
        {
            // Linear 需要：x, weight, [bias]
            if (node->attributes.find("weight") != node->attributes.end())
            {
                auto &weight_attr = node->attributes["weight"];
                if (!weight_attr.data.empty() && !weight_attr.shape.empty())
                {
                    std::vector<size_t> weight_shape_vec;
                    for (int dim : weight_attr.shape)
                    {
                        weight_shape_vec.push_back(static_cast<size_t>(dim));
                    }
                    Shape weight_shape(weight_shape_vec);

                    // 确定设备类型（从输入 tensor 获取）
                    Device device(DeviceType::kCPU);
                    if (input_tensors.size() > 0)
                    {
                        device = input_tensors[0].device();
                    }
#ifdef WITH_CUDA
                    else if (cuda::is_available())
                    {
                        device = Device(DeviceType::kCUDA, 0);
                    }
#endif

                    Tensor weight_tensor(weight_attr.data, weight_shape, dtype(DataType::kFloat32).device(device));

                    // 在 x 之后插入 weight
                    if (input_tensors.size() > 0)
                    {
                        Tensor x = input_tensors[0];
                        input_tensors.clear();
                        input_tensors.push_back(x);
                        input_tensors.push_back(weight_tensor);

                        // 如果有偏置，也添加
                        if (node->attributes.find("bias") != node->attributes.end())
                        {
                            auto &bias_attr = node->attributes["bias"];
                            if (!bias_attr.data.empty() && !bias_attr.shape.empty())
                            {
                                std::vector<size_t> bias_shape_vec;
                                for (int dim : bias_attr.shape)
                                {
                                    bias_shape_vec.push_back(static_cast<size_t>(dim));
                                }
                                Shape bias_shape(bias_shape_vec);
                                Tensor bias_tensor(bias_attr.data, bias_shape,
                                                   dtype(DataType::kFloat32).device(device));
                                input_tensors.push_back(bias_tensor);
                            }
                        }
                    }
                }
            }
        }

        // 执行当前节点算子，得到 output_tensors，再传播到下游（含 pnnx.Output）
        if (!input_tensors.empty())
        {
            logd("Executing node {} (type: {}) with {} input tensor(s)", node->name, node->type, input_tensors.size());
            try
            {
                auto start_time = std::chrono::high_resolution_clock::now();

                node->output_tensors = (*node->op)(input_tensors);

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

                logi("Layer name: {}\tlayer type: {}\ttime cost: {}us", node->name, node->type, duration.count());

                // 将本节点 output_tensors 按 output_names 写入所有下游的 input_tensors（含 pnnx.Output，即最终模型输出）
                auto propagate_start = std::chrono::high_resolution_clock::now();
                propagate_outputs(node);
                auto propagate_end = std::chrono::high_resolution_clock::now();
                auto propagate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(propagate_end - propagate_start);
                if (propagate_duration.count() > 10)
                {
                    logw("propagate_outputs for node {} took {}ms", node->name, propagate_duration.count());
                }
            }
            catch (const std::exception &e)
            {
                loge("Error executing node {} ({}): {}", node->name, node->type, e.what());
                throw;
            }
        }
        else
        {
            // 如果没有输入，但仍然有输出（不应该发生，但为了安全起见）
            if (!node->output_tensors.empty())
            {
                propagate_outputs(node);
            }
            else if (node->type == "models.yolo.Detect")
            {
                logw("Detect node has no input tensors");
            }
        }
    }

    // 输出总执行时间
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
    logi("Forward completed: {} nodes executed in {}us", total_nodes, total_duration.count());

#ifdef WITH_CUDA
    // 确保 CUDA 操作完成（同步所有 CUDA 流）
    if (cuda::is_available())
    {
        cudaDeviceSynchronize();
        logi("CUDA synchronization completed");
    }
#endif
}

// 用户调用 get_outputs("pnnx_output_0") 时进入。输出节点类型为 pnnx.Output，其 input_names 指向上游；
// 返回的即这些上游的 output_tensors（forward 中已通过 propagate_outputs 写入 output_node->input_tensors）。
// yolov5_infer 中 outputs[0] 为检测头结果，形状如 [N, num_boxes, 5+num_classes]，后续做阈值与 NMS。
// 注意：输出节点（pnnx.Output）是特殊节点：input_count>=1（有 input_names，指向上游输出），output_count=0（output_names 为空），
// 不参与拓扑排序和执行，没有对应的 Operator（op 为空），其 input_tensors 由上游节点的 propagate_outputs 填充。
std::vector<Tensor> PNNXGraph::get_outputs(const std::string &output_name) const
{
    logi("Getting outputs for: {}", output_name);

    if (graph_state_ != GraphState::Complete)
    {
        THROW_RUNTIME_ERROR("Graph must be built before getting outputs");
    }

    auto it = node_map_.find(output_name);
    if (it == node_map_.end())
    {
        THROW_RUNTIME_ERROR("Output node not found: {}", output_name);
    }

    auto output_node = it->second;

    // 输出节点不计算，只"收集"上游输出；其 input_names 即模型最终输出的张量名
    // 输出节点没有 output_names（output_count=0），只有 input_names（input_count>=1），作为图的出口
    std::vector<Tensor> outputs;

    // 遍历输出节点的所有输入名称（每个对应一个上游输出）
    for (const auto &input_name : output_node->input_names)
    {
        // 优先从 output_node 的 input_tensors 取（forward 里 propagate_outputs 已写入）
        auto input_it = output_node->input_tensors.find(input_name);
        if (input_it != output_node->input_tensors.end())
        {
            outputs.push_back(input_it->second);
            continue;
        }

        // 如果 input_tensors 中没有，从上游节点的 output_tensors 中获取
        bool found = false;
        for (const auto &node : nodes_)
        {
            // 跳过输入输出节点
            if (node->type == "pnnx.Input" || node->type == "pnnx.Output")
            {
                continue;
            }

            for (size_t i = 0; i < node->output_names.size(); ++i)
            {
                if (node->output_names[i] == input_name)
                {
                    if (i < node->output_tensors.size())
                    {
                        outputs.push_back(node->output_tensors[i]);
                        found = true;
                        break;
                    }
                    else
                    {
                        // 调试：输出名称匹配但输出 Tensor 为空
                        logw("Warning: Node {} has output name '{}' but output_tensors[{}] is empty", node->name,
                             input_name, i);
                    }
                }
            }
            if (found) 
            {
                break;
            }
        }

        if (!found)
        {
            // 输出未找到的警告（仅在调试模式下）
            logw("Output node input '{}' not found", input_name);
        }
    }

    logi("Retrieved {} output tensor(s)", outputs.size());
    return outputs;
}

bool PNNXGraph::is_input_node(const std::string &name) const
{
    auto it = node_map_.find(name);
    if (it != node_map_.end())
    {
        return it->second->type == "pnnx.Input";
    }
    return false;
}

bool PNNXGraph::is_output_node(const std::string &name) const
{
    auto it = node_map_.find(name);
    if (it != node_map_.end())
    {
        return it->second->type == "pnnx.Output";
    }
    return false;
}

void PNNXGraph::topological_sort_dfs(std::shared_ptr<PNNXNode> node,
                                     std::map<std::string, bool> &visited,
                                     std::vector<std::shared_ptr<PNNXNode>> &sorted)
{
    // DFS 版本的拓扑排序（备用实现）
    visited[node->name] = true;

    // 处理所有下游节点
    for (const auto &output_name : node->output_names)
    {
        for (auto &other_node : nodes_)
        {
            for (const auto &input_name : other_node->input_names)
            {
                if (input_name == output_name && !visited[other_node->name])
                {
                    topological_sort_dfs(other_node, visited, sorted);
                }
            }
        }
    }

    sorted.push_back(node);
}

// 将本节点的 output_tensors 按 output_names 写入所有以该名为 input_name 的下游节点的 input_tensors（含 pnnx.Output）
// 输出节点（pnnx.Output）通过此函数接收上游输出，其 input_tensors 在 get_outputs() 中被收集返回
void PNNXGraph::propagate_outputs(std::shared_ptr<PNNXNode> node)
{
    for (size_t i = 0; i < node->output_names.size(); ++i)
    {
        const std::string &output_name = node->output_names[i];

        if (i >= node->output_tensors.size())
        {
            continue;  // 没有对应的输出 Tensor
        }

        // 找到使用这个输出的下游节点（遍历所有节点，检查它们的输入名称）
        // 注意：这里不能使用映射，因为一个输出可能被多个节点使用
        for (auto &downstream_node : nodes_)
        {
            // 跳过输入节点
            if (downstream_node->type == "pnnx.Input")
            {
                continue;
            }

            // 检查下游节点的所有输入名称
            for (const auto &input_name : downstream_node->input_names)
            {
                if (input_name == output_name)
                {
                    // 使用 input_name 作为 key，因为这是下游节点期望的输入名称
                    downstream_node->input_tensors[input_name] = node->output_tensors[i];
                }
            }
        }
    }
}

// 建立 输出张量名 -> (产生该输出的节点, 输出索引)，forward 收集输入时可据此快速查上游，避免全图遍历（若未调用则 forward 内会退化为遍历 nodes_ 查找）
void PNNXGraph::build_output_name_map()
{
    output_name_map_.clear();

    for (auto &node : nodes_)
    {
        // 跳过输入输出节点
        if (node->type == "pnnx.Input" || node->type == "pnnx.Output")
        {
            continue;
        }

        // 为每个输出名称建立映射
        for (size_t i = 0; i < node->output_names.size(); ++i)
        {
            const std::string &output_name = node->output_names[i];
            output_name_map_[output_name]  = std::make_pair(node, i);
        }
    }
}

}  // namespace pnnx
}  // namespace origin
