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
 * @details 加载和管理 PNNX 模型，执行推理
 */
class PNNXGraph
{
public:
    /**
     * @brief 构造函数
     * @param param_path .param 文件路径
     * @param bin_path .bin 文件路径
     */
    PNNXGraph(const std::string &param_path, const std::string &bin_path);

    /**
     * @brief 构建计算图
     * @details 解析模型文件，创建算子，建立连接关系，进行拓扑排序
     */
    void build();

    /**
     * @brief 设置输入
     * @param input_name 输入节点名称（如 "pnnx_input_0"）
     * @param inputs 输入 Tensor 列表
     */
    void set_inputs(const std::string &input_name, const std::vector<Tensor> &inputs);

    /**
     * @brief 执行前向推理
     * @param debug 是否输出调试信息
     */
    void forward(bool debug = false);

    /**
     * @brief 获取输出
     * @param output_name 输出节点名称（如 "pnnx_output_0"）
     * @return 输出 Tensor 列表
     */
    std::vector<Tensor> get_outputs(const std::string &output_name) const;

private:
    std::string param_path_;
    std::string bin_path_;
    std::vector<std::shared_ptr<PNNXNode>> nodes_;
    std::map<std::string, std::shared_ptr<PNNXNode>> node_map_;  // 名称到节点的映射
    std::map<std::string, std::pair<std::shared_ptr<PNNXNode>, size_t>>
        output_name_map_;  // 输出名称到(节点, 索引)的映射

    // 输入输出节点
    std::vector<std::shared_ptr<PNNXNode>> input_nodes_;
    std::vector<std::shared_ptr<PNNXNode>> output_nodes_;

    // 图状态
    enum class GraphState
    {
        NeedInit,
        NeedBuild,
        Complete
    };
    GraphState graph_state_ = GraphState::NeedInit;

    /**
     * @brief 初始化：解析模型文件
     */
    bool init();

    /**
     * @brief 创建节点连接关系
     */
    void create_node_relations();

    /**
     * @brief 拓扑排序，确定执行顺序
     */
    void topological_sort();

    /**
     * @brief 递归拓扑排序辅助函数
     */
    void topological_sort_dfs(std::shared_ptr<PNNXNode> node,
                              std::map<std::string, bool> &visited,
                              std::vector<std::shared_ptr<PNNXNode>> &sorted);

    /**
     * @brief 检查是否为输入节点
     */
    bool is_input_node(const std::string &name) const;

    /**
     * @brief 检查是否为输出节点
     */
    bool is_output_node(const std::string &name) const;

    /**
     * @brief 传播节点输出到下游节点输入
     */
    void propagate_outputs(std::shared_ptr<PNNXNode> node);

    /**
     * @brief 构建输出名称到节点的映射，加速查找
     */
    void build_output_name_map();
};

}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_GRAPH_H__
