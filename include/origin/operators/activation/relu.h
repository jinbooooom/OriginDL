#ifndef __ORIGIN_DL_RELU_H__
#define __ORIGIN_DL_RELU_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief ReLU 激活函数算子
 *
 * 计算 ReLU 激活函数，用于神经网络
 * 公式：ReLU(x) = max(0, x)
 */
class ReLU : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    void forward_inplace(Tensor &input0, const Tensor &input1) override;
};

/**
 * @brief 计算张量的 ReLU 激活函数
 *
 * @param x 输入张量
 * @return ReLU 激活结果，ReLU(x) = max(0, x)
 */
extern Tensor relu(const Tensor &x);

// 原地操作函数
extern void relu_(Tensor &x);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_RELU_H__
