#ifndef __ORIGIN_DL_CONV2D_LAYER_H__
#define __ORIGIN_DL_CONV2D_LAYER_H__

#include "../../core/operator.h"
#include "../../core/parameter.h"
#include "../../core/tensor.h"
#include "../layer.h"
#include <utility>

namespace origin
{
namespace nn
{

/**
 * @brief 二维卷积层
 * @details 实现 y = conv2d(x, W) + b
 */
class Conv2d : public Layer
{
private:
    Parameter weight_;  // 权重参数，形状 (out_channels, in_channels, kernel_h, kernel_w)
    Parameter bias_;    // 偏置参数，形状 (out_channels,)
    int in_channels_;
    int out_channels_;
    std::pair<int, int> kernel_size_;  // (kernel_h, kernel_w)
    std::pair<int, int> stride_;       // (stride_h, stride_w)
    std::pair<int, int> pad_;          // (pad_h, pad_w)
    bool use_bias_;

public:
    /**
     * @brief 构造函数
     * @param in_channels 输入通道数
     * @param out_channels 输出通道数
     * @param kernel_size 卷积核大小 (kernel_h, kernel_w)
     * @param stride 步长，默认为 (1, 1)
     * @param pad 填充，默认为 (0, 0)
     * @param bias 是否使用偏置，默认为true
     */
    Conv2d(int in_channels, int out_channels, std::pair<int, int> kernel_size, 
           std::pair<int, int> stride = {1, 1}, std::pair<int, int> pad = {0, 0}, bool bias = true);

    /**
     * @brief 构造函数（单值版本，kernel_size, stride, pad 为单个值）
     * @param in_channels 输入通道数
     * @param out_channels 输出通道数
     * @param kernel_size 卷积核大小（正方形）
     * @param stride 步长，默认为 1
     * @param pad 填充，默认为 0
     * @param bias 是否使用偏置，默认为true
     */
    Conv2d(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int pad = 0, bool bias = true);

    /**
     * @brief 前向传播
     * @param input 输入张量，形状 (N, C, H, W)
     * @return 输出张量，形状 (N, OC, OH, OW)
     */
    Tensor forward(const Tensor &input) override;

    /**
     * @brief 参数访问
     * @return 权重参数
     */
    Parameter *weight() { return &weight_; }

    /**
     * @brief 参数访问
     * @return 偏置参数
     */
    Parameter *bias() { return use_bias_ ? &bias_ : nullptr; }

    /**
     * @brief 重置参数
     */
    void reset_parameters();

private:
    /**
     * @brief 初始化权重参数（Kaiming初始化，适合ReLU）
     */
    Parameter init_weight();

    /**
     * @brief 初始化偏置参数
     */
    Parameter init_bias();

    /**
     * @brief 初始化参数（用于reset_parameters）
     */
    void init_parameters();
};

}  // namespace nn
}  // namespace origin

#endif  // __ORIGIN_DL_CONV2D_LAYER_H__
