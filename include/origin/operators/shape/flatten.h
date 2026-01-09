#ifndef __ORIGIN_DL_FLATTEN_OPERATOR_H__
#define __ORIGIN_DL_FLATTEN_OPERATOR_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief Flatten 算子
 * @details 将输入张量从 start_dim 到 end_dim 的所有维度展平为一个维度
 */
class FlattenOp : public Operator
{
public:
    int start_dim_;  // 起始维度，默认为 1
    int end_dim_;    // 结束维度，默认为 -1（最后一个维度）

    FlattenOp(int start_dim = 1, int end_dim = -1) : start_dim_(start_dim), end_dim_(end_dim) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 函数式接口：Flatten 算子
 * @details 将输入张量从 start_dim 到 end_dim 的所有维度展平为一个维度
 * 参考 PyTorch 的 torch.flatten(input, start_dim=0, end_dim=-1)
 * 
 * @param x 输入张量
 * @param start_dim 起始维度，默认为 1（保留 batch 维度）
 * @param end_dim 结束维度，默认为 -1（最后一个维度）
 * @return 展平后的张量
 */
Tensor flatten(const Tensor &x, int start_dim = 1, int end_dim = -1);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_FLATTEN_OPERATOR_H__

