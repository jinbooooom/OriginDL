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
    int out_features_;  // 输出特征数
    bool use_bias_;     // 是否使用偏置

    LinearOp(int in_features, int out_features, bool use_bias = true)
        : in_features_(in_features), out_features_(out_features), use_bias_(use_bias)
    {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 函数式接口：Linear 算子
 * @param x 输入张量
 * @param weight 权重张量 (out_features, in_features)
 * @param bias 偏置张量 (out_features, optional)
 * @param in_features 输入特征数
 * @param out_features 输出特征数
 * @param use_bias 是否使用偏置，默认 true
 * @return 输出张量 (N, out_features)
 */
Tensor custom_linear(const Tensor &x,
                     const Tensor &weight,
                     const Tensor &bias,
                     int in_features,
                     int out_features,
                     bool use_bias = true);

/**
 * @brief 函数式接口：Linear 算子（无偏置版本）
 * @param x 输入张量
 * @param weight 权重张量 (out_features, in_features)
 * @param in_features 输入特征数
 * @param out_features 输出特征数
 * @return 输出张量 (N, out_features)
 */
Tensor custom_linear(const Tensor &x, const Tensor &weight, int in_features, int out_features);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_LINEAR_OPERATOR_H__
