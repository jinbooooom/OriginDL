#ifndef __ORIGIN_DL_FLATTEN_LAYER_H__
#define __ORIGIN_DL_FLATTEN_LAYER_H__

#include "../../core/operator.h"
#include "../../core/tensor.h"
#include "../layer.h"

namespace origin
{
namespace nn
{

/**
 * @brief Flatten 层
 * @details 将输入张量展平，从 start_dim 到 end_dim 的所有维度展平为一个维度
 */
class Flatten : public Layer
{
private:
    int start_dim_;  // 起始维度，默认为 1
    int end_dim_;    // 结束维度，默认为 -1（最后一个维度）

public:
    /**
     * @brief 构造函数
     * @param start_dim 起始维度，默认为 1
     * @param end_dim 结束维度，默认为 -1（最后一个维度）
     */
    Flatten(int start_dim = 1, int end_dim = -1);

    /**
     * @brief 前向传播
     * @param input 输入张量
     * @return 展平后的张量
     */
    Tensor forward(const Tensor &input) override;
};

}  // namespace nn
}  // namespace origin

#endif  // __ORIGIN_DL_FLATTEN_LAYER_H__
