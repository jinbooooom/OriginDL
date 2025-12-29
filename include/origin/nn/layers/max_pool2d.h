#ifndef __ORIGIN_DL_MAX_POOL2D_LAYER_H__
#define __ORIGIN_DL_MAX_POOL2D_LAYER_H__

#include "../../core/operator.h"
#include "../../core/tensor.h"
#include "../layer.h"
#include <utility>

namespace origin
{
namespace nn
{

/**
 * @brief 二维最大池化层
 * @details 实现二维最大池化操作
 */
class MaxPool2d : public Layer
{
private:
    std::pair<int, int> kernel_size_;  // (kernel_h, kernel_w)
    std::pair<int, int> stride_;       // (stride_h, stride_w)
    std::pair<int, int> pad_;          // (pad_h, pad_w)

public:
    /**
     * @brief 构造函数
     * @param kernel_size 池化核大小 (kernel_h, kernel_w)
     * @param stride 步长，默认为 (0, 0)，表示使用 kernel_size
     * @param pad 填充，默认为 (0, 0)
     */
    MaxPool2d(std::pair<int, int> kernel_size, 
              std::pair<int, int> stride = {0, 0}, 
              std::pair<int, int> pad = {0, 0});

    /**
     * @brief 构造函数（单值版本）
     * @param kernel_size 池化核大小（正方形）
     * @param stride 步长，默认为 0，表示使用 kernel_size
     * @param pad 填充，默认为 0
     */
    MaxPool2d(int kernel_size, int stride = 0, int pad = 0);

    /**
     * @brief 前向传播
     * @param input 输入张量，形状 (N, C, H, W)
     * @return 输出张量，形状 (N, C, OH, OW)
     */
    Tensor forward(const Tensor &input) override;
};

}  // namespace nn
}  // namespace origin

#endif  // __ORIGIN_DL_MAX_POOL2D_LAYER_H__

