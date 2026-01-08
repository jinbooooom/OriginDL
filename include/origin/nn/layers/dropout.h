#ifndef __ORIGIN_DL_DROPOUT_LAYER_H__
#define __ORIGIN_DL_DROPOUT_LAYER_H__

#include "../../core/tensor.h"
#include "../layer.h"

namespace origin
{
namespace nn
{

/**
 * @brief Dropout 层
 * @details 训练时随机将部分神经元输出置为 0，防止过拟合
 */
class Dropout : public Layer
{
private:
    float p_;  // dropout 概率

public:
    /**
     * @brief 构造函数
     * @param p dropout 概率，默认为 0.5
     */
    Dropout(float p = 0.5f);

    /**
     * @brief 前向传播
     * @param input 输入张量
     * @return 输出张量
     */
    Tensor forward(const Tensor &input) override;
};

}  // namespace nn
}  // namespace origin

#endif  // __ORIGIN_DL_DROPOUT_LAYER_H__
