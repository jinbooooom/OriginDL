#ifndef __ORIGIN_DL_SUB_H__
#define __ORIGIN_DL_SUB_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Sub : public Operator
{
public:
    Shape shape0_;
    Shape shape1_;

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    void forward_inplace(Tensor &input0, const Tensor &input1) override;
};

extern Tensor sub(const std::vector<Tensor> &xs);
extern Tensor sub(const Tensor &lhs, const Tensor &rhs);

// 原地操作函数
extern void sub_(Tensor &lhs, const Tensor &rhs);

}  // namespace functional

// 运算符重载放在 origin 命名空间下
Tensor operator-(const Tensor &lhs, const Tensor &rhs);
Tensor operator-(const Tensor &lhs, const Scalar &rhs);
Tensor operator-(const Scalar &lhs, const Tensor &rhs);

}  // namespace origin

#endif  // __ORIGIN_DL_SUB_H__

