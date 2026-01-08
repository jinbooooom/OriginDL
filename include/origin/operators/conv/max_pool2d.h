#ifndef __ORIGIN_DL_MAX_POOL2D_H__
#define __ORIGIN_DL_MAX_POOL2D_H__

#include <vector>
#include "../../core/operator.h"
#include "../../utils/conv_utils.h"

namespace origin
{

/**
 * @brief MaxPool2d 算子：二维最大池化操作
 *
 * 输入：
 * - x: 输入张量，形状 (N, C, H, W)
 *
 * 输出：
 * - y: 形状为 (N, C, OH, OW) 的张量
 */
class MaxPool2d : public Operator
{
public:
    std::pair<int, int> kernel_size_;
    std::pair<int, int> stride_;
    std::pair<int, int> pad_;

    // 保存前向传播的索引（用于反向传播）
    std::vector<size_t> indices_;  // 形状：(N*C*OH*OW,)
    Shape indices_shape_;          // (N, C, OH, OW)

    MaxPool2d(std::pair<int, int> kernel_size, std::pair<int, int> stride = {0, 0}, std::pair<int, int> pad = {0, 0})
        : kernel_size_(kernel_size), stride_(stride), pad_(pad)
    {
        // 如果stride为0，则使用kernel_size作为stride
        if (stride_.first == 0)
        {
            stride_.first = kernel_size_.first;
        }
        if (stride_.second == 0)
        {
            stride_.second = kernel_size_.second;
        }
    }

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief MaxPool2d 函数
 * @param x 输入张量，形状 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长，默认等于kernel_size
 * @param pad 填充，默认 0
 * @return 形状为 (N, C, OH, OW) 的张量
 */
Tensor max_pool2d(const Tensor &x,
                  std::pair<int, int> kernel_size,
                  std::pair<int, int> stride = {0, 0},
                  std::pair<int, int> pad    = {0, 0});

Tensor max_pool2d(const Tensor &x, int kernel_size, int stride = 0, int pad = 0);

}  // namespace origin

#endif  // __ORIGIN_DL_MAX_POOL2D_H__
