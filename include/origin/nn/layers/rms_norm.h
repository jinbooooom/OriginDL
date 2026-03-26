#ifndef __ORIGIN_DL_RMS_NORM_LAYER_H__
#define __ORIGIN_DL_RMS_NORM_LAYER_H__

#include "../../core/parameter.h"
#include "../../core/tensor.h"
#include "../layer.h"

namespace origin
{
namespace nn
{

/**
 * @brief RMSNorm 归一化层
 * @details 对输入进行 RMSNorm 归一化，公式：output = gamma * input / sqrt(mean(input^2) + eps)
 *          RMSNorm 与 LayerNorm 的区别在于：RMSNorm 不包含 centering（减去均值）操作
 */
class RMSNorm : public Layer
{
private:
    int normalized_shape_;  // 归一化的维度（通常是最后一个维度的大小）
    float eps_;             // 数值稳定性参数
    Parameter gamma_;       // 缩放参数，形状 (normalized_shape,)

public:
    /**
     * @brief 构造函数
     * @param normalized_shape 归一化的特征数
     * @param eps 数值稳定性参数，默认为 1e-5
     */
    RMSNorm(int normalized_shape, float eps = 1e-5f);

    /**
     * @brief 前向传播
     * @param input 输入张量，形状 (..., normalized_shape)
     * @return 输出张量，形状与输入相同
     */
    Tensor forward(const Tensor &input) override;

    /**
     * @brief 参数访问
     * @return gamma 参数
     */
    Parameter *weight() { return &gamma_; }

    /**
     * @brief 重置参数
     */
    void reset_parameters();
};

}  // namespace nn
}  // namespace origin

#endif  // __ORIGIN_DL_RMS_NORM_LAYER_H__
