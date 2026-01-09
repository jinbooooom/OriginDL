#ifndef __ORIGIN_DL_ADAPTIVE_AVG_POOL2D_H__
#define __ORIGIN_DL_ADAPTIVE_AVG_POOL2D_H__

#include "../../core/operator.h"

namespace origin
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

}  // namespace origin

#endif  // __ORIGIN_DL_ADAPTIVE_AVG_POOL2D_H__

