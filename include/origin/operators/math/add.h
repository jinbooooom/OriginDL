#ifndef __ORIGIN_DL_ADD_H__
#define __ORIGIN_DL_ADD_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Add : public Operator
{
public:
    Shape shape0_;
    Shape shape1_;

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor add(const std::vector<Tensor> &xs);
extern Tensor add(const Tensor &lhs, const Tensor &rhs);

}  // namespace functional

// 运算符重载放在 origin 命名空间下
Tensor operator+(const Tensor &lhs, const Tensor &rhs);
Tensor operator+(const Tensor &lhs, const Scalar &rhs);
Tensor operator+(const Scalar &lhs, const Tensor &rhs);

}  // namespace origin

#endif  // __ORIGIN_DL_ADD_H__

