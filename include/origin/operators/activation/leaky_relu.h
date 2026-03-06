#ifndef __ORIGIN_DL_LEAKY_RELU_H__
#define __ORIGIN_DL_LEAKY_RELU_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief LeakyReLU 激活函数算子
 *
 * 计算 LeakyReLU 激活函数，用于神经网络
 * 公式：LeakyReLU(x) = x (x>0) or LeayReLU(x) = ax (x<0)
 */
class LeakyReLU : public Operator
{
public:
    LeakyReLU(float alpha = 0.1f) : alpha_(alpha) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

private:
    float alpha_;

    Tensor mask_;  // 保存 (x > 0) 的mask，用于反向传播
};

/**
 * @brief 计算张量的 LeakyReLU 激活函数
 *
 * @param x 输入张量
 * @return LeakyReLU 激活结果。LeakyReLU(x) = x (x>0) or LeayReLU(x) = ax (x<0)
 */
extern Tensor leaky_relu(const Tensor &x, float alpha);

}  // namespace functional
}  // namespace origin

#endif