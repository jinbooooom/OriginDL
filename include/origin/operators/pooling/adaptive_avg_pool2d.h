#ifndef __ORIGIN_DL_ADAPTIVE_AVG_POOL2D_H__
#define __ORIGIN_DL_ADAPTIVE_AVG_POOL2D_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief AdaptiveAvgPool2d 算子
 * @details 自适应平均池化，将输入调整为指定的输出尺寸
 */
class AdaptiveAvgPool2d : public Operator
{
public:
    std::pair<int, int> output_size_;  // 输出尺寸 (H, W)

    AdaptiveAvgPool2d(std::pair<int, int> output_size) : output_size_(output_size) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief AdaptiveAvgPool2d 函数
 * @param x 输入张量，形状 (N, C, H, W)
 * @param output_size 输出尺寸 (OH, OW)
 * @return 形状为 (N, C, OH, OW) 的张量
 */
Tensor adaptive_avg_pool2d(const Tensor &x, std::pair<int, int> output_size);

Tensor adaptive_avg_pool2d(const Tensor &x, int output_size);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_ADAPTIVE_AVG_POOL2D_H__

