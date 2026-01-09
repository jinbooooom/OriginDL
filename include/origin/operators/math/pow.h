#ifndef __ORIGIN_DL_POW_H__
#define __ORIGIN_DL_POW_H__

#include "../../core/operator.h"

namespace origin
{

class Pow : public Operator
{
public:
    // 支持多种类型的指数构造函数
    Pow(Scalar n) : exponent_(n){};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    Scalar exponent_;  // 幂函数的指数，支持多种数值类型
};

extern Tensor pow(const Tensor &base, const Scalar &exponent);        // 支持标量指数
extern Tensor operator^(const Tensor &base, const Scalar &exponent);  // 支持标量指数

}  // namespace origin

#endif  // __ORIGIN_DL_POW_H__

