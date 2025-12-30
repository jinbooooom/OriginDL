#ifndef __ORIGIN_DL_BATCH_NORM2D_LAYER_H__
#define __ORIGIN_DL_BATCH_NORM2D_LAYER_H__

#include "../../core/parameter.h"
#include "../../core/tensor.h"
#include "../layer.h"

namespace origin
{
namespace nn
{

/**
 * @brief 二维批归一化层
 * @details 对输入 (N, C, H, W) 进行批归一化
 */
class BatchNorm2d : public Layer
{
private:
    Parameter gamma_;      // 缩放参数，形状 (num_features,)
    Parameter beta_;       // 偏移参数，形状 (num_features,)
    Tensor running_mean_;  // 移动平均均值，形状 (num_features,)
    Tensor running_var_;   // 移动平均方差，形状 (num_features,)
    int num_features_;
    float eps_;
    float momentum_;

public:
    /**
     * @brief 构造函数
     * @param num_features 特征数（通道数）
     * @param eps 数值稳定性参数，默认为 1e-5
     * @param momentum 移动平均的动量，默认为 0.1
     */
    BatchNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f);

    /**
     * @brief 前向传播
     * @param input 输入张量，形状 (N, C, H, W)
     * @return 输出张量，形状 (N, C, H, W)
     */
    Tensor forward(const Tensor &input) override;

    /**
     * @brief 参数访问
     * @return gamma 参数
     */
    Parameter *weight() { return &gamma_; }

    /**
     * @brief 参数访问
     * @return beta 参数
     */
    Parameter *bias() { return &beta_; }

    /**
     * @brief 重置参数
     */
    void reset_parameters();
};

}  // namespace nn
}  // namespace origin

#endif  // __ORIGIN_DL_BATCH_NORM2D_LAYER_H__

