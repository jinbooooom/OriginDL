#ifndef __ORIGIN_DL_NEG_H__
#define __ORIGIN_DL_NEG_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Neg : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

Tensor neg(const std::vector<Tensor> &xs);
Tensor neg(const Tensor &x);

}  // namespace functional

// 运算符重载放在 origin 命名空间下
Tensor operator-(const Tensor &x);

}  // namespace origin

#endif  // __ORIGIN_DL_NEG_H__

