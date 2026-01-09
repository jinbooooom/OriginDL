#ifndef __ORIGIN_DL_SIGMOID_H__
#define __ORIGIN_DL_SIGMOID_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief Sigmoid 激活函数算子
 *
 * 计算 Sigmoid 激活函数，用于神经网络
 * 公式：sigmoid(x) = 1 / (1 + exp(-x))
 */
class Sigmoid : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算张量的 Sigmoid 激活函数
 *
 * @param x 输入张量
 * @return Sigmoid 激活结果，sigmoid(x) = 1 / (1 + exp(-x))
 */
extern Tensor sigmoid(const Tensor &x);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_SIGMOID_H__

