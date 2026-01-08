#ifndef __ORIGIN_DL_SILU_H__
#define __ORIGIN_DL_SILU_H__

#include "../../core/operator.h"

namespace origin
{

/**
 * @brief SiLU 激活函数算子
 * @details SiLU (Sigmoid Linear Unit) = x * sigmoid(x)
 */
class SiLU : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算张量的 SiLU 激活函数
 * @param x 输入张量
 * @return SiLU 激活结果，SiLU(x) = x * sigmoid(x)
 */
extern Tensor silu(const Tensor &x);

}  // namespace origin

#endif  // __ORIGIN_DL_SILU_H__

