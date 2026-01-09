#ifndef __ORIGIN_DL_MUL_H__
#define __ORIGIN_DL_MUL_H__

#include "../../core/operator.h"

namespace origin
{

class Mul : public Operator
{
public:
    Shape shape0_;
    Shape shape1_;

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor mul(const std::vector<Tensor> &xs);
extern Tensor mul(const Tensor &lhs, const Tensor &rhs);
Tensor operator*(const Tensor &lhs, const Tensor &rhs);
Tensor operator*(const Tensor &lhs, const Scalar &rhs);
Tensor operator*(const Scalar &lhs, const Tensor &rhs);

}  // namespace origin

#endif  // __ORIGIN_DL_MUL_H__

