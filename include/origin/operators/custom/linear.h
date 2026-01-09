#ifndef __ORIGIN_DL_LINEAR_OPERATOR_H__
#define __ORIGIN_DL_LINEAR_OPERATOR_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief Linear 算子（全连接层）
 * @details 实现 y = x * W^T + b
 * 输入：x (N, in_features), weight (out_features, in_features), bias (out_features, optional)
 * 输出：y (N, out_features)
 */
class LinearOp : public Operator
{
public:
    int in_features_;   // 输入特征数
    int out_features_;   // 输出特征数
    bool use_bias_;     // 是否使用偏置

    LinearOp(int in_features, int out_features, bool use_bias = true)
        : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_LINEAR_OPERATOR_H__

