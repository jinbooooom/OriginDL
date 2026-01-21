#ifndef __ORIGIN_DL_POW_H__
#define __ORIGIN_DL_POW_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Pow : public Operator
{
public:
    // 支持多种类型的指数构造函数
    Pow(Scalar n) : exponent_(n){};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    void forward_inplace(Tensor &input0, const Tensor &input1) override;

    Scalar exponent_;  // 幂函数的指数，支持多种数值类型
};

extern Tensor pow(const Tensor &base, const Scalar &exponent);  // 支持标量指数

// 原地操作函数
extern void pow_(Tensor &x, const Scalar &exponent);

}  // namespace functional

Tensor operator^(const Tensor &base, const Scalar &exponent);  // 支持标量指数

// 就地操作运算符重载
Tensor &operator^=(Tensor &x, const Scalar &exponent);

}  // namespace origin

#endif  // __ORIGIN_DL_POW_H__
