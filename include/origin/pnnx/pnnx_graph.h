/**
 * @file pnnx_graph.h
 * @brief PNNX 计算图对外接口
 * @details 本头文件定义 PNNXGraph，是使用 PNNX 模型推理的主入口。用户通过构造(param,
 * bin)、build()、set_inputs()、forward()、get_outputs() 完成加载与推理； 内部依赖 PNNXParser 解析、PNNXNode
 * 表示节点、OperatorMapper 创建算子。实现见 src/pnnx/pnnx_graph.cpp。
 */
#ifndef __ORIGIN_DL_PNNX_GRAPH_H__
#define __ORIGIN_DL_PNNX_GRAPH_H__

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "../core/tensor.h"
#include "pnnx_node.h"

namespace origin
{
namespace pnnx
{

/**
 * @brief PNNX 计算图
 * @details
 * 本类负责加载和管理 PNNX 格式的模型，并执行推理。典型用法（参见 tests/example/yolo/yolov5_infer.cpp）：
 * 1) 构造：PNNXGraph graph(param_path, bin_path) —— 仅保存路径，不读文件；
 * 2) 构建：graph.build() —— 内部依次执行：init() 解析 param/bin 得到节点列表并建立 node_map_，
 *    create_node_relations() 建立节点间连接，topological_sort() 确定 execution_order，
 *    再为每个非 Input/Output 节点通过 OperatorMapper 创建对应的 origin Operator；
 * 3) 设置输入：graph.set_inputs("pnnx_input_0", {input}) —— 将用户 Tensor 写入输入节点的 output_tensors，
 *    并通过 propagate_outputs 填到下游节点的 input_tensors；
 * 4) 执行：graph.forward() —— 按 execution_order 依次执行各节点的 op，每节点执行后 propagate_outputs；
 * 5) 取输出：graph.get_outputs("pnnx_output_0") —— 从输出节点的 input_tensors（即上游的最终输出）收集并返回。
 * 输入/输出节点名称通常为 "pnnx_input_0"、"pnnx_output_0"，由 PNNX 导出格式决定。
 */
class PNNXGraph
{
public:
    /**
     * @brief 构造函数
     * @param param_path .param 文件路径（描述图结构、形状、属性元信息）
     * @param bin_path .bin 文件路径（存储权重二进制数据）；仅保存路径，实际读取在 build() -> init() 中进行
     */
    PNNXGraph(const std::string &param_path, const std::string &bin_path);

    /**
     * @brief 构建计算图
     * @details 若未初始化则先 init()（PNNXParser::parse 得到 nodes_ 与 node_map_），
     * 再 create_node_relations()、topological_sort() 为节点赋 execution_order，
     * 最后对非 pnnx.Input/pnnx.Output 节点调用 OperatorMapper::create_operator 填充 node->op。
     * 调用后 graph_state_ 变为 Complete，方可 set_inputs/forward/get_outputs。
     */
    void build();

    /**
     * @brief 设置输入
     * @param input_name 输入节点名称（如 "pnnx_input_0"）
     * @param inputs 输入 Tensor 列表（数量须与输入节点的 output_names 数量一致）
     * @details 将 inputs 写入该输入节点的 output_tensors，并调用 propagate_outputs 传播到所有下游的 input_tensors
     */
    void set_inputs(const std::string &input_name, const std::vector<Tensor> &inputs);

    /**
     * @brief 执行前向推理
     * @details 按 execution_order 排序后依次执行各节点：收集 input_tensors（含从上游 output 或 output_name_map_
     * 查找）， 对 Conv2d/Linear 等从 node->attributes 注入 weight/bias，执行 (*node->op)(input_tensors) 得到
     * output_tensors， 再 propagate_outputs 写入下游的 input_tensors。输出节点的“输入”即为模型最终输出。
     */
    void forward();

    /**
     * @brief 获取输出
     * @param output_name 输出节点名称（如 "pnnx_output_0"）
     * @return 输出 Tensor 列表（输出节点的每个 input_name 对应一个上游 tensor）
     * @details 输出节点类型为 pnnx.Output，其 input_names 指向上游输出；从该节点的 input_tensors 或上游的
     * output_tensors 收集返回
     */
    std::vector<Tensor> get_outputs(const std::string &output_name) const;

private:
    std::string param_path_;                                     ///< .param 文件路径
    std::string bin_path_;                                       ///< .bin 文件路径
    std::vector<std::shared_ptr<PNNXNode>> nodes_;               ///< 解析得到的全部节点（含 Input/Output）
    std::map<std::string, std::shared_ptr<PNNXNode>> node_map_;  ///< 节点名 -> 节点，用于按名查找输入/输出节点
    std::map<std::string, std::pair<std::shared_ptr<PNNXNode>, size_t>>
        output_name_map_;  ///< 中间张量名 -> (产生该输出的节点, 输出索引)，forward 时加速上游查找

    std::vector<std::shared_ptr<PNNXNode>> input_nodes_;   ///< 输入节点列表（类型 pnnx.Input）
    std::vector<std::shared_ptr<PNNXNode>> output_nodes_;  ///< 输出节点列表（类型 pnnx.Output）
    // 预先计算好的执行计划：所有需要执行的节点（排除 pnnx.Input/pnnx.Output），按拓扑排序后的 execution_order 排好序，
    // build() 阶段构建一次，后续每次 forward 直接按本列表顺序执行，避免重复排序和去重。
    std::vector<std::shared_ptr<PNNXNode>> execution_plan_;

    /** 图状态：NeedInit 未解析；NeedBuild 已 init 待 build；Complete 可 set_inputs/forward/get_outputs */
    enum class GraphState
    {
        NeedInit,
        NeedBuild,
        Complete
    };
    GraphState graph_state_ = GraphState::NeedInit;

    /**
     * @brief 初始化：解析模型文件
     * @return 成功与否；内部调用 PNNXParser::parse(param_path_, bin_path_) 得到 nodes_，并填充 node_map_
     */
    bool init();

    /**
     * @brief 拓扑排序，确定执行顺序
     * @details 通过 output_names 与 input_names 匹配建立节点间连接关系（构建邻接表），
     * 按图依赖计算入度，BFS 得到顺序，为每个非 Input/Output 节点设置 execution_order
     */
    void topological_sort();

    /**
     * @brief 递归拓扑排序辅助函数（DFS 版本，备用）
     */
    void topological_sort_dfs(std::shared_ptr<PNNXNode> node,
                              std::map<std::string, bool> &visited,
                              std::vector<std::shared_ptr<PNNXNode>> &sorted);

    /** @brief 检查 node_map_ 中是否存在该名且类型为 pnnx.Input */
    bool is_input_node(const std::string &name) const;

    /** @brief 检查 node_map_ 中是否存在该名且类型为 pnnx.Output */
    bool is_output_node(const std::string &name) const;

    /**
     * @brief 传播节点输出到下游节点输入
     * @details 将该节点的 output_tensors 按 output_names 写入所有下游节点的 input_tensors（按 input_name 匹配）
     */
    void propagate_outputs(std::shared_ptr<PNNXNode> node);

    /**
     * @brief 构建 output_name_map_：每个非 Input/Output 节点的每个 output_names[i] -> (node, i)，用于 forward
     * 时快速查上游
     */
    void build_output_name_map();
};

}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_GRAPH_H__
