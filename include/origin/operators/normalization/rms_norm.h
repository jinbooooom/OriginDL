#ifndef __ORIGIN_DL_RMS_NORM_H__
#define __ORIGIN_DL_RMS_NORM_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class RMSNorm : public Operator
{
public:
    float eps_;

    RMSNorm(float eps);

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

private:
    Tensor saved_rms_;  // 保存 RMS 值用于反向传播
};

/**
 * @brief 函数式接口：RMSNorm 算子
 * @details RMSNorm 归一化，公式：output = gamma * input / sqrt(mean(input^2) + eps)
 *          与 PyTorch 实现对齐
 *
 * @param x 输入张量，形状 (..., normalized_shape)
 * @param gamma 缩放参数 (weight)，形状为 (normalized_shape,)
 * @param eps 数值稳定性参数，默认 1e-5
 * @return 输出张量，形状与输入相同
 */
Tensor rms_norm(const Tensor &x, const Tensor &gamma, float eps = 1e-5f);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_RMS_NORM_H__
