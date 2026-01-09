#ifndef __ORIGIN_DL_FLATTEN_OPERATOR_H__
#define __ORIGIN_DL_FLATTEN_OPERATOR_H__

#include "../../core/operator.h"

namespace origin
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

}  // namespace origin

#endif  // __ORIGIN_DL_FLATTEN_OPERATOR_H__

