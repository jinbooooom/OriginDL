// 简化的 PNNXGraph 实现
#include "origin/pnnx/pnnx_graph.h"
#include "origin/pnnx/pnnx_parser.h"
#include "origin/pnnx/operator_mapper.h"
#include "origin/utils/exception.h"
#include "origin/utils/log.h"
#include "origin/core/config.h"
#ifdef WITH_CUDA
#include "origin/cuda/cuda.h"
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>

namespace origin
{
namespace pnnx
{

PNNXGraph::PNNXGraph(const std::string &param_path, const std::string &bin_path)
    : param_path_(param_path), bin_path_(bin_path)
{
}

bool PNNXGraph::init()
{
    if (param_path_.empty() || bin_path_.empty())
    {
        THROW_RUNTIME_ERROR("Param path or bin path is empty");
        return false;
    }
    
    // 解析模型文件
    nodes_ = PNNXParser::parse(param_path_, bin_path_);
    
    if (nodes_.empty())
    {
        THROW_RUNTIME_ERROR("Failed to parse PNNX model or model is empty");
        return false;
    }
    
    // 建立节点映射
    for (auto &node : nodes_)
    {
        node_map_[node->name] = node;
    }
    
    graph_state_ = GraphState::NeedBuild;
    return true;
}

void PNNXGraph::build()
{
    if (graph_state_ == GraphState::Complete)
    {
        return;  // 已经构建完成
    }
    
    if (graph_state_ == GraphState::NeedInit)
    {
        bool success = init();
        if (!success || graph_state_ == GraphState::NeedInit)
        {
            THROW_RUNTIME_ERROR("Failed to initialize graph");
        }
    }
    
    // 创建节点连接关系
    create_node_relations();
    
    // 拓扑排序
    topological_sort();
    
    // 为每个节点创建对应的 Operator（除了 input/output 节点）
    for (auto &node : nodes_)
    {
        if (node->type != "pnnx.Input" && node->type != "pnnx.Output")
        {
            node->op = OperatorMapper::create_operator(node);
            if (!node->op)
            {
                THROW_RUNTIME_ERROR("Failed to create operator for node: {} type: {}", 
                                    node->name, node->type);
            }
        }
    }
    
    graph_state_ = GraphState::Complete;
}

void PNNXGraph::create_node_relations()
{
    // 建立节点之间的连接关系
    for (auto &current_node : nodes_)
    {
        // 找到当前节点的输出连接到的下游节点
        for (const auto &output_name : current_node->output_names)
        {
            // 查找使用这个输出作为输入的节点
            for (auto &other_node : nodes_)
            {
                if (other_node == current_node) continue;
                
                // 检查 other_node 的输入是否包含 output_name
                for (const auto &input_name : other_node->input_names)
                {
                    if (input_name == output_name)
                    {
                        // 建立连接关系（可以通过 node_map 快速查找）
                        // 这里暂时只记录，执行时再查找
                    }
                }
            }
        }
    }
}

void PNNXGraph::topological_sort()
{
    // 简单的拓扑排序实现
    std::map<std::string, int> in_degree;  // 每个节点的入度
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
                if (other_node == node) continue;
                
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

void PNNXGraph::set_inputs(const std::string &input_name, const std::vector<Tensor> &inputs)
{
    if (graph_state_ != GraphState::Complete)
    {
        THROW_RUNTIME_ERROR("Graph must be built before setting inputs");
    }
    
    auto it = node_map_.find(input_name);
    if (it == node_map_.end())
    {
        THROW_RUNTIME_ERROR("Input node not found: {}", input_name);
    }
    
    auto input_node = it->second;
    
    // 设置输入节点的输出（输入节点的输出名称就是它的输出）
    if (input_node->output_names.size() == inputs.size())
    {
        input_node->output_tensors = inputs;
        
        // 将输入 Tensor 传递给下游节点
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

void PNNXGraph::forward(bool debug)
{
    if (graph_state_ != GraphState::Complete)
    {
        THROW_RUNTIME_ERROR("Graph must be built before forward");
    }
    
    // 记录总开始时间
    auto total_start_time = std::chrono::high_resolution_clock::now();
    
    // 按拓扑排序顺序执行节点（只执行有 execution_order 的节点）
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
    size_t total_nodes = sorted_nodes.size();
    size_t executed_count = 0;
    std::set<std::string> executed_nodes;  // 用于检测重复执行
    
    for (auto &node : sorted_nodes)
    {
        // 检查是否重复执行
        if (executed_nodes.find(node->name) != executed_nodes.end())
        {
            std::cerr << "ERROR: Node " << node->name << " is being executed multiple times!" << std::endl;
            THROW_RUNTIME_ERROR("Node {} is being executed multiple times", node->name);
        }
        executed_nodes.insert(node->name);
        
        executed_count++;
        if (executed_count % 10 == 0 || executed_count == total_nodes)
        {
            std::cerr << "Progress: " << executed_count << "/" << total_nodes 
                      << " nodes executed (current: " << node->name << ")" << std::endl;
        }
        
        if (!node->op)
        {
            THROW_RUNTIME_ERROR("Operator not created for node: {}", node->name);
        }
        
        // 收集输入 Tensor
        std::vector<Tensor> input_tensors;
        if (executed_count > 10 && executed_count <= 15)
        {
            logi("Collecting inputs for node {} (type: {})", node->name, node->type);
        }
        // 调试：打印 torch.cat_1 的输入收集过程
        if (node->name == "torch.cat_1")
        {
            std::cout << "\n=== DEBUG: origindl Collecting inputs for torch.cat_1 ===" << std::endl;
            std::cout << "Input names: ";
            for (const auto &name : node->input_names) {
                std::cout << name << " ";
            }
            std::cout << std::endl;
            std::cout << "Number of input_names: " << node->input_names.size() << std::endl;
        }
        for (const auto &input_name : node->input_names)
        {
            auto it = node->input_tensors.find(input_name);
            if (it != node->input_tensors.end())
            {
                input_tensors.push_back(it->second);
            }
            else
            {
                // 从上游节点获取输出（使用映射加速查找）
                auto output_it = output_name_map_.find(input_name);
                if (output_it != output_name_map_.end())
                {
                    auto &prev_node = output_it->second.first;
                    size_t output_idx = output_it->second.second;
                    
                    if (output_idx < prev_node->output_tensors.size())
                    {
                        input_tensors.push_back(prev_node->output_tensors[output_idx]);
                        node->input_tensors[input_name] = prev_node->output_tensors[output_idx];
                        
                        // 调试：打印 torch.cat_1 的输入来源
                        if (node->name == "torch.cat_1")
                        {
                            auto &tensor = prev_node->output_tensors[output_idx];
                            std::cout << "  Input from node: " << prev_node->name 
                                      << ", output_idx: " << output_idx
                                      << ", shape: " << tensor.shape().to_string() << std::endl;
                        }
                    }
                }
            }
        }
        
        // 调试：打印 torch.cat_1 收集到的所有输入
        if (node->name == "torch.cat_1")
        {
            std::cout << "Collected " << input_tensors.size() << " input tensors:" << std::endl;
            for (size_t i = 0; i < input_tensors.size(); ++i) {
                std::cout << "  Input " << i << " shape: " << input_tensors[i].shape().to_string() << std::endl;
            }
            std::cout << "===================================\n" << std::endl;
        }
        
        
        // 对于需要权重的算子（如 Conv2d, Linear），添加权重和偏置
        if (node->type == "nn.Conv2d" || node->type == "nn.Linear")
        {
            // Conv2d 需要：x, weight, [bias]
            // 注意：input_tensors[0] 是输入 x，需要在其后插入 weight 和 bias
            
            // 从 attributes 中获取权重
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
                    
                    Tensor weight_tensor(weight_attr.data, weight_shape, 
                                        dtype(DataType::kFloat32).device(device));
                    
                    // 调试：检查第一个卷积层的输入和权重
                    // 检查 model.3.conv 的输入
                    if (node->name == "model.3.conv" && input_tensors.size() > 0)
                    {
                        auto &x = input_tensors[0];
                        auto x_shape = x.shape();
                        auto x_data = x.to_vector<float>();
                        
                        std::cout << "\n=== DEBUG: origindl model.3.conv Input ===" << std::endl;
                        std::cout << "Input shape: " << x_shape.to_string() << std::endl;
                        std::cout << "Input data stats: min=" << *std::min_element(x_data.begin(), x_data.end())
                                  << ", max=" << *std::max_element(x_data.begin(), x_data.end())
                                  << ", mean=" << std::accumulate(x_data.begin(), x_data.end(), 0.0f) / x_data.size() << std::endl;
                        std::cout << "Input first 10 values: ";
                        for (size_t i = 0; i < std::min(size_t(10), x_data.size()); ++i) {
                            std::cout << x_data[i] << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "===================================\n" << std::endl;
                    }
                    if (node->name == "model.0.conv" && input_tensors.size() > 0)
                    {
                        // 检查权重布局：打印前几个权重值，分析是否是列主序
                        std::cout << "[Weight Layout Check] First conv layer weight:" << std::endl;
                        std::cout << "  Shape: (";
                        for (size_t i = 0; i < weight_attr.shape.size(); ++i) {
                            std::cout << weight_attr.shape[i];
                            if (i < weight_attr.shape.size() - 1) std::cout << ", ";
                        }
                        std::cout << ")" << std::endl;
                        
                        if (weight_attr.shape.size() == 4) {
                            int OC = weight_attr.shape[0];
                            int IC = weight_attr.shape[1];
                            int KH = weight_attr.shape[2];
                            int KW = weight_attr.shape[3];
                            
                            // 行主序索引: (0,0,0,0) 和 (0,0,0,1) 应该相邻
                            int row_idx1 = 0 * (IC * KH * KW) + 0 * (KH * KW) + 0 * KW + 0;
                            int row_idx2 = 0 * (IC * KH * KW) + 0 * (KH * KW) + 0 * KW + 1;
                            
                            // 列主序索引: (0,0,0,0) 和 (1,0,0,0) 应该相邻
                            int col_idx1 = 0 * (KH * IC * OC) + 0 * (IC * OC) + 0 * OC + 0;
                            int col_idx2 = 0 * (KH * IC * OC) + 0 * (IC * OC) + 0 * OC + 1;
                            
                            std::cout << "  Row-major check: idx(" << row_idx1 << ")=" << weight_attr.data[row_idx1]
                                      << ", idx(" << row_idx2 << ")=" << weight_attr.data[row_idx2] << std::endl;
                            std::cout << "  Col-major check: idx(" << col_idx1 << ")=" << weight_attr.data[col_idx1]
                                      << ", idx(" << col_idx2 << ")=" << weight_attr.data[col_idx2] << std::endl;
                            
                            // 如果 row_idx2 - row_idx1 == 1，说明是行主序
                            // 如果 col_idx2 - col_idx1 == 1，说明是列主序
                            if (row_idx2 - row_idx1 == 1) {
                                std::cout << "  -> Likely ROW-MAJOR (adjacent kw indices are consecutive)" << std::endl;
                            } else if (col_idx2 - col_idx1 == 1) {
                                std::cout << "  -> Likely COLUMN-MAJOR (adjacent oc indices are consecutive)" << std::endl;
                            }
                        }
                    }
                    if (node->name == "model.0.conv" && input_tensors.size() > 0)
                    {
                        auto &x = input_tensors[0];
                        auto x_shape = x.shape();
                        auto x_data = x.to_vector<float>();
                        
                        std::cout << "\n=== DEBUG: First Conv2d Layer (model.0.conv) ===" << std::endl;
                        std::cout << "Input shape: " << x_shape.to_string() << std::endl;
                        std::cout << "Input data stats: min=" << *std::min_element(x_data.begin(), x_data.end())
                                  << ", max=" << *std::max_element(x_data.begin(), x_data.end())
                                  << ", mean=" << std::accumulate(x_data.begin(), x_data.end(), 0.0f) / x_data.size() << std::endl;
                        std::cout << "Input first 10 values: ";
                        for (size_t i = 0; i < std::min(size_t(10), x_data.size()); ++i) {
                            std::cout << x_data[i] << " ";
                        }
                        std::cout << std::endl;
                        
                        std::cout << "Weight shape: " << weight_shape.to_string() << std::endl;
                        auto weight_data = weight_attr.data;
                        std::cout << "Weight data stats: min=" << *std::min_element(weight_data.begin(), weight_data.end())
                                  << ", max=" << *std::max_element(weight_data.begin(), weight_data.end())
                                  << ", mean=" << std::accumulate(weight_data.begin(), weight_data.end(), 0.0f) / weight_data.size() << std::endl;
                        std::cout << "Weight first 10 values: ";
                        for (size_t i = 0; i < std::min(size_t(10), weight_data.size()); ++i) {
                            std::cout << weight_data[i] << " ";
                        }
                        std::cout << std::endl;
                    }
                    
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
                    
                    Tensor weight_tensor(weight_attr.data, weight_shape, 
                                        dtype(DataType::kFloat32).device(device));
                    
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
        
        // 执行算子
        if (!input_tensors.empty())
        {
            if (executed_count > 10 && executed_count <= 15)
            {
                logi("Executing node {} (type: {}) with {} input tensor(s)", 
                     node->name, node->type, input_tensors.size());
            }
            try
            {
                // 记录执行开始时间
                auto start_time = std::chrono::high_resolution_clock::now();
                
                node->output_tensors = (*node->op)(input_tensors);
                
                // 调试：输出关键层的输出数据
                if (debug && !node->output_tensors.empty())
                {
                    // pnnx.Expression 操作（add）
                    if (node->name == "pnnx_expr_19" || node->name == "pnnx_expr_21")
                    {
                        // 打印输入
                        if (input_tensors.size() >= 2) {
                            auto input0_data = input_tensors[0].to_vector<float>();
                            auto input1_data = input_tensors[1].to_vector<float>();
                            std::cout << "\n=== DEBUG: origindl Expression Input (pnnx_expr_19) ===" << std::endl;
                            std::cout << "Input 0 shape: " << input_tensors[0].shape().to_string() << std::endl;
                            std::cout << "Input 0 first 10 values: ";
                            for (size_t i = 0; i < std::min(size_t(10), input0_data.size()); ++i) {
                                std::cout << input0_data[i] << " ";
                            }
                            std::cout << std::endl;
                            std::cout << "Input 1 shape: " << input_tensors[1].shape().to_string() << std::endl;
                            std::cout << "Input 1 first 10 values: ";
                            for (size_t i = 0; i < std::min(size_t(10), input1_data.size()); ++i) {
                                std::cout << input1_data[i] << " ";
                            }
                            std::cout << std::endl;
                        }
                        
                        auto &output = node->output_tensors[0];
                        auto output_shape = output.shape();
                        auto output_data = output.to_vector<float>();
                        
                        std::cout << "\n=== DEBUG: origindl Expression Output (" << node->name << ") ===" << std::endl;
                        std::cout << "Output shape: " << output_shape.to_string() << std::endl;
                        std::cout << "Output data stats: min=" << *std::min_element(output_data.begin(), output_data.end())
                                  << ", max=" << *std::max_element(output_data.begin(), output_data.end())
                                  << ", mean=" << std::accumulate(output_data.begin(), output_data.end(), 0.0f) / output_data.size() << std::endl;
                        std::cout << "Output first 10 values: ";
                        for (size_t i = 0; i < std::min(size_t(10), output_data.size()); ++i) {
                            std::cout << output_data[i] << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "===================================\n" << std::endl;
                    }
                    // torch.cat 操作
                    else if (node->type == "torch.cat" && (node->name.find("cat") != std::string::npos || node->name.find("torch.cat") != std::string::npos))
                    {
                        auto &output = node->output_tensors[0];
                        auto output_shape = output.shape();
                        auto output_data = output.to_vector<float>();
                        
                        std::cout << "\n=== DEBUG: origindl Cat Output (" << node->name << ") ===" << std::endl;
                        std::cout << "Output shape: " << output_shape.to_string() << std::endl;
                        std::cout << "Output data stats: min=" << *std::min_element(output_data.begin(), output_data.end())
                                  << ", max=" << *std::max_element(output_data.begin(), output_data.end())
                                  << ", mean=" << std::accumulate(output_data.begin(), output_data.end(), 0.0f) / output_data.size() << std::endl;
                        std::cout << "Output first 10 values: ";
                        for (size_t i = 0; i < std::min(size_t(10), output_data.size()); ++i) {
                            std::cout << output_data[i] << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "===================================\n" << std::endl;
                    }
                    // 卷积层
                    else if (node->name == "model.0.conv" || node->name == "model.1.conv" || 
                        node->name == "model.2.cv1.conv" || node->name == "model.2.cv2.conv" || 
                        node->name == "model.2.cv3.conv" || node->name == "model.3.conv" ||
                        node->name == "model.4.cv1.conv" || node->name == "model.4.cv2.conv" ||
                        node->name == "model.4.m.1.cv1.conv" || node->name == "model.4.m.1.cv2.conv" ||
                        node->name == "model.5.conv" ||
                        node->name == "model.6.cv1.conv" || node->name == "model.7.conv" ||
                        node->name == "model.8.cv1.conv" || node->name == "model.9.conv" ||
                        node->name == "model.10.cv1.conv" || node->name == "model.11.conv")
                    {
                        auto &output = node->output_tensors[0];
                        auto output_shape = output.shape();
                        auto output_data = output.to_vector<float>();
                        
                        std::cout << "\n=== DEBUG: origindl Conv2d Output (" << node->name << ") ===" << std::endl;
                        std::cout << "Output shape: " << output_shape.to_string() << std::endl;
                        std::cout << "Output data stats: min=" << *std::min_element(output_data.begin(), output_data.end())
                                  << ", max=" << *std::max_element(output_data.begin(), output_data.end())
                                  << ", mean=" << std::accumulate(output_data.begin(), output_data.end(), 0.0f) / output_data.size() << std::endl;
                        std::cout << "Output first 10 values: ";
                        for (size_t i = 0; i < std::min(size_t(10), output_data.size()); ++i) {
                            std::cout << output_data[i] << " ";
                        }
                        std::cout << std::endl;
                        
                        // 打印特定位置的数据（用于与 KuiperInferGitee 对比）
                        // origindl shape: {N, C, H, W} = {4, 16, 160, 160}，行主序
                        // 打印 batch 0 的特定位置：(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 0, 1)
                        if (output_shape.size() == 4) {
                            size_t N = output_shape[0];
                            size_t C = output_shape[1];
                            size_t H = output_shape[2];
                            size_t W = output_shape[3];
                            std::cout << "Output specific positions (batch 0, row-major indexing):" << std::endl;
                            std::vector<std::tuple<size_t, size_t, size_t, size_t>> positions = {
                                {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 1, 0, 1},
                                {0, 0, 0, 2}, {0, 0, 1, 1}, {0, 1, 0, 2}, {0, 2, 0, 0}, {0, 2, 0, 1}
                            };
                            for (const auto& pos : positions) {
                                size_t n = std::get<0>(pos);
                                size_t c = std::get<1>(pos);
                                size_t h = std::get<2>(pos);
                                size_t w = std::get<3>(pos);
                                if (n < N && c < C && h < H && w < W) {
                                    // 行主序索引: n * (C*H*W) + c * (H*W) + h * W + w
                                    size_t idx = n * (C * H * W) + c * (H * W) + h * W + w;
                                    std::cout << "  [" << n << "," << c << "," << h << "," << w << "] = " << output_data[idx] << std::endl;
                                }
                            }
                        }
                        std::cout << "===================================\n" << std::endl;
                    }
                    // 激活函数（SiLU）
                    else if (node->name == "model.0.act" || node->name == "model.1.act" ||
                             node->name == "model.2.cv1.act" || node->name == "model.2.cv2.act" ||
                             node->name == "model.2.cv3.act" || node->name == "model.3.act" ||
                             node->name == "model.4.cv1.act" || node->name == "model.4.cv2.act" ||
                             node->name == "model.4.m.1.cv1.act" || node->name == "model.4.m.1.cv2.act" ||
                             node->name == "model.5.act" ||
                             node->name == "model.6.cv1.act" || node->name == "model.7.act" ||
                             node->name == "model.8.cv1.act" || node->name == "model.9.act" ||
                             node->name == "model.10.cv1.act" || node->name == "model.11.act" ||
                             node->type == "nn.SiLU" || node->type == "F.silu")
                    {
                        // 打印输入（SiLU 的输入应该是前一层卷积的输出）
                        if (!input_tensors.empty()) {
                            auto input_data = input_tensors[0].to_vector<float>();
                            auto input_shape = input_tensors[0].shape();
                            std::cout << "\n=== DEBUG: origindl SiLU Input (" << node->name << ", type: " << node->type << ") ===" << std::endl;
                            std::cout << "Input shape: " << input_shape.to_string() << std::endl;
                            std::cout << "Input data stats: min=" << *std::min_element(input_data.begin(), input_data.end())
                                      << ", max=" << *std::max_element(input_data.begin(), input_data.end())
                                      << ", mean=" << std::accumulate(input_data.begin(), input_data.end(), 0.0f) / input_data.size() << std::endl;
                            std::cout << "Input first 10 values: ";
                            for (size_t i = 0; i < std::min(size_t(10), input_data.size()); ++i) {
                                std::cout << input_data[i] << " ";
                            }
                            std::cout << std::endl;
                        }
                        
                        auto &output = node->output_tensors[0];
                        auto output_shape = output.shape();
                        auto output_data = output.to_vector<float>();
                        
                        std::cout << "\n=== DEBUG: origindl SiLU Output (" << node->name << ", type: " << node->type << ") ===" << std::endl;
                        std::cout << "Output shape: " << output_shape.to_string() << std::endl;
                        std::cout << "Output data stats: min=" << *std::min_element(output_data.begin(), output_data.end())
                                  << ", max=" << *std::max_element(output_data.begin(), output_data.end())
                                  << ", mean=" << std::accumulate(output_data.begin(), output_data.end(), 0.0f) / output_data.size() << std::endl;
                        std::cout << "Output first 10 values: ";
                        for (size_t i = 0; i < std::min(size_t(10), output_data.size()); ++i) {
                            std::cout << output_data[i] << " ";
                        }
                        std::cout << std::endl;
                        
                        // 验证 SiLU 计算：SiLU(x) = x * sigmoid(x)
                        // 对于第一层，对比输入和输出，验证 SiLU 是否正确计算
                        if (node->name == "model.0.act" && !input_tensors.empty()) {
                            auto input_data = input_tensors[0].to_vector<float>();
                            std::cout << "SiLU verification (first 5 values):" << std::endl;
                            for (size_t i = 0; i < std::min(size_t(5), output_data.size()); ++i) {
                                float x = input_data[i];
                                float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
                                float expected_silu = x * sigmoid_x;
                                float actual_silu = output_data[i];
                                std::cout << "  [" << i << "] x=" << x << ", sigmoid(x)=" << sigmoid_x 
                                          << ", expected SiLU=" << expected_silu 
                                          << ", actual SiLU=" << actual_silu
                                          << ", diff=" << std::abs(expected_silu - actual_silu) << std::endl;
                            }
                        }
                        std::cout << "===================================\n" << std::endl;
                    }
                    // YoloDetect 层输入
                    else if (node->name.find("yolo") != std::string::npos || node->name.find("Detect") != std::string::npos)
                    {
                        if (!input_tensors.empty())
                        {
                            auto &input = input_tensors[0];
                            auto input_shape = input.shape();
                            auto input_data = input.to_vector<float>();
                            
                            std::cout << "\n=== DEBUG: origindl YoloDetect Input (" << node->name << ") ===" << std::endl;
                            std::cout << "Input shape: " << input_shape.to_string() << std::endl;
                            std::cout << "Input data stats: min=" << *std::min_element(input_data.begin(), input_data.end())
                                      << ", max=" << *std::max_element(input_data.begin(), input_data.end())
                                      << ", mean=" << std::accumulate(input_data.begin(), input_data.end(), 0.0f) / input_data.size() << std::endl;
                            std::cout << "Input first 10 values: ";
                            for (size_t i = 0; i < std::min(size_t(10), input_data.size()); ++i) {
                                std::cout << input_data[i] << " ";
                            }
                            std::cout << std::endl;
                            std::cout << "===================================\n" << std::endl;
                        }
                    }
                }
                
                // 记录执行结束时间
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                // 输出每个层的执行时间
                logi("Layer name: {}\tlayer type: {}\ttime cost: {}ms", 
                     node->name, node->type, duration.count());
                
                // 将输出传播到下游节点（包括输出节点）
                auto propagate_start = std::chrono::high_resolution_clock::now();
                propagate_outputs(node);
                auto propagate_end = std::chrono::high_resolution_clock::now();
                auto propagate_duration = std::chrono::duration_cast<std::chrono::milliseconds>(propagate_end - propagate_start);
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
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
    logi("Forward completed: {} nodes executed in {}ms", total_nodes, total_duration.count());
    
#ifdef WITH_CUDA
    // 确保 CUDA 操作完成（同步所有 CUDA 流）
    if (cuda::is_available())
    {
        cudaDeviceSynchronize();
        logi("CUDA synchronization completed");
    }
#endif
}

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
    
    // 返回输出节点的输入 Tensor（因为输出节点只是收集上游的输出）
    std::vector<Tensor> outputs;
    
    // 遍历输出节点的所有输入名称
    for (const auto &input_name : output_node->input_names)
    {
        // 首先尝试从 output_node 的 input_tensors 中获取
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
                        std::cerr << "Warning: Node " << node->name << " has output name '" << input_name 
                                  << "' but output_tensors[" << i << "] is empty" << std::endl;
                    }
                }
            }
            if (found) break;
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

void PNNXGraph::propagate_outputs(std::shared_ptr<PNNXNode> node)
{
    // 将节点的输出传播到下游节点（包括输出节点）
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
            output_name_map_[output_name] = std::make_pair(node, i);
        }
    }
}

}  // namespace pnnx
}  // namespace origin

