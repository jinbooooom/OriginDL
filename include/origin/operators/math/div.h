#ifndef __ORIGIN_DL_DIV_H__
#define __ORIGIN_DL_DIV_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Div : public Operator
{
public:
    Shape shape0_;
    Shape shape1_;

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    void forward_inplace(Tensor &input0, const Tensor &input1) override;
};

extern Tensor div(const std::vector<Tensor> &xs);
extern Tensor div(const Tensor &lhs, const Tensor &rhs);

// 原地操作函数
extern void div_(Tensor &lhs, const Tensor &rhs);

}  // namespace functional

Tensor operator/(const Tensor &lhs, const Tensor &rhs);
Tensor operator/(const Tensor &lhs, const Scalar &rhs);
Tensor operator/(const Scalar &lhs, const Tensor &rhs);

// 就地操作运算符重载
Tensor &operator/=(Tensor &lhs, const Tensor &rhs);
Tensor &operator/=(Tensor &lhs, const Scalar &rhs);

}  // namespace origin

#endif  // __ORIGIN_DL_DIV_H__
