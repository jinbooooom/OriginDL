/**
 * @file operator_mapper.h
 * @brief PNNX 算子类型到 origin Operator 的映射器声明
 * @details 声明 OperatorMapper，根据 PNNXNode::type（如 nn.Conv2d、nn.SiLU、models.yolo.Detect）创建对应的 origin
 * Operator 实例。 在 PNNXGraph::build() 中为每个计算节点调用 create_operator(node)；实现见
 * src/pnnx/operator_mapper.cpp。
 */
#ifndef __ORIGIN_DL_PNNX_OPERATOR_MAPPER_H__
#define __ORIGIN_DL_PNNX_OPERATOR_MAPPER_H__

#include <memory>
#include "../core/operator.h"
#include "pnnx_node.h"

namespace origin
{
namespace pnnx
{

/**
 * @brief PNNX 算子映射器
 * @details
 * 将 PNNX 节点类型（node->type）映射到 origin 的 Operator 实现。在 PNNXGraph::build() 中，
 * 对每个非 pnnx.Input/pnnx.Output 节点调用 create_operator(node)，根据 type 分发到对应的 create_xxx，
 * 从 node->params、node->attributes 读取参数和权重，构造并返回 std::shared_ptr<Operator>。
 * 支持的 type 示例：nn.Conv2d、nn.SiLU、nn.ReLU、nn.AdaptiveAvgPool2d、nn.Flatten、nn.Linear、
 * nn.Upsample、nn.MaxPool2d、F.linear、torch.cat、models.yolo.Detect 等；未支持的 type 会导致创建失败。
 */
class OperatorMapper
{
public:
    /**
     * @brief 根据节点类型创建对应的 origin Operator
     * @param node 已由 PNNXParser 解析好的节点（含 params、attributes）
     * @return 可执行的 Operator，forward 时以 (*node->op)(input_tensors) 调用
     */
    static std::shared_ptr<Operator> create_operator(std::shared_ptr<PNNXNode> node);

private:
    /** @brief nn.Conv2d：从 params 取 stride/padding/dilation 等，从 attributes 取 weight、bias */
    static std::shared_ptr<Operator> create_conv2d(std::shared_ptr<PNNXNode> node);

    /** @brief nn.SiLU（Swish）：无额外参数 */
    static std::shared_ptr<Operator> create_silu(std::shared_ptr<PNNXNode> node);

    /** @brief nn.ReLU：无额外参数 */
    static std::shared_ptr<Operator> create_relu(std::shared_ptr<PNNXNode> node);

    /** @brief nn.AdaptiveAvgPool2d：从 params 取 output_size */
    static std::shared_ptr<Operator> create_adaptive_avg_pool2d(std::shared_ptr<PNNXNode> node);

    /** @brief nn.Flatten：从 params 取 start_dim/end_dim 等 */
    static std::shared_ptr<Operator> create_flatten(std::shared_ptr<PNNXNode> node);

    /** @brief nn.Linear / F.linear：从 attributes 取 weight、bias */
    static std::shared_ptr<Operator> create_linear(std::shared_ptr<PNNXNode> node);

    /** @brief nn.Upsample：从 params 取 size/scale_factor、mode 等 */
    static std::shared_ptr<Operator> create_upsample(std::shared_ptr<PNNXNode> node);

    /** @brief nn.MaxPool2d：从 params 取 kernel_size、stride、padding */
    static std::shared_ptr<Operator> create_max_pool2d(std::shared_ptr<PNNXNode> node);

    /** @brief Expression（如 add、mul 等表达式）：从 params 取表达式字符串并解析 */
    static std::shared_ptr<Operator> create_expression(std::shared_ptr<PNNXNode> node);

    /** @brief torch.cat：从 params 取 dim */
    static std::shared_ptr<Operator> create_cat(std::shared_ptr<PNNXNode> node);

    /** @brief models.yolo.Detect：YOLO 检测头，从 params 取 anchor/stride 等，用于后处理 */
    static std::shared_ptr<Operator> create_yolo_detect(std::shared_ptr<PNNXNode> node);

    /** @brief 从 node->params 中取整型参数，缺省时返回 default_value */
    static int get_int_param(const std::map<std::string, Parameter> &params, const std::string &key, int default_value);

    /** @brief 从 node->params 中取 (int,int) 参数（如 kernel_size=(2,2)） */
    static std::pair<int, int> get_int_pair_param(const std::map<std::string, Parameter> &params,
                                                  const std::string &key,
                                                  std::pair<int, int> default_value);

    /** @brief 将 Attribute（shape+data）转为 origin Tensor，用于 weight/bias 注入 */
    static Tensor load_weight_tensor(const Attribute &attr);
};

}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_OPERATOR_MAPPER_H__
