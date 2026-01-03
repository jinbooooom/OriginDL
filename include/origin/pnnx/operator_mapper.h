#ifndef __ORIGIN_DL_PNNX_OPERATOR_MAPPER_H__
#define __ORIGIN_DL_PNNX_OPERATOR_MAPPER_H__

#include <memory>
#include "pnnx_node.h"
#include "../core/operator.h"

namespace origin
{
namespace pnnx
{

/**
 * @brief PNNX 算子映射器
 * @details 将 PNNX 算子类型映射到 origindl 的 Operator
 */
class OperatorMapper
{
public:
    /**
     * @brief 创建算子
     * @param node PNNX 节点
     * @return 对应的 origindl Operator
     */
    static std::shared_ptr<Operator> create_operator(std::shared_ptr<PNNXNode> node);

private:
    /**
     * @brief 创建 Conv2d 算子
     */
    static std::shared_ptr<Operator> create_conv2d(std::shared_ptr<PNNXNode> node);

    /**
     * @brief 创建 SiLU 算子
     */
    static std::shared_ptr<Operator> create_silu(std::shared_ptr<PNNXNode> node);

    /**
     * @brief 创建 Upsample 算子
     */
    static std::shared_ptr<Operator> create_upsample(std::shared_ptr<PNNXNode> node);

    /**
     * @brief 创建 MaxPool2d 算子
     */
    static std::shared_ptr<Operator> create_max_pool2d(std::shared_ptr<PNNXNode> node);

    /**
     * @brief 创建 Expression 算子（如 add, mul 等）
     */
    static std::shared_ptr<Operator> create_expression(std::shared_ptr<PNNXNode> node);

    /**
     * @brief 创建 cat 算子
     */
    static std::shared_ptr<Operator> create_cat(std::shared_ptr<PNNXNode> node);

    /**
     * @brief 从参数中获取整数值
     */
    static int get_int_param(const std::map<std::string, Parameter> &params,
                              const std::string &key, int default_value);

    /**
     * @brief 从参数中获取整数对
     */
    static std::pair<int, int> get_int_pair_param(const std::map<std::string, Parameter> &params,
                                                   const std::string &key,
                                                   std::pair<int, int> default_value);

    /**
     * @brief 从属性中加载权重 Tensor
     */
    static Tensor load_weight_tensor(const Attribute &attr);
};

}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_OPERATOR_MAPPER_H__

