#ifndef __ORIGIN_DL_RELU_LAYER_H__
#define __ORIGIN_DL_RELU_LAYER_H__

#include "../../core/operator.h"
#include "../../core/tensor.h"
#include "../layer.h"

namespace origin
{
namespace nn
{

/**
 * @brief ReLU 激活函数层
 * @details 实现 y = ReLU(x) = max(0, x)
 */
class ReLU : public Layer
{
public:
    /**
     * @brief 构造函数
     */
    ReLU() = default;

    /**
     * @brief 前向传播
     * @param input 输入张量
     * @return 输出张量
     */
    Tensor forward(const Tensor &input) override;
};

}  // namespace nn
}  // namespace origin

#endif  // __ORIGIN_DL_RELU_LAYER_H__

