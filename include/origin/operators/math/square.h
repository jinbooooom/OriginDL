#ifndef __ORIGIN_DL_SQUARE_H__
#define __ORIGIN_DL_SQUARE_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Square : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    void forward_inplace(Tensor &input0, const Tensor &input1) override;
};

extern Tensor square(const Tensor &x);

// 原地操作函数
extern void square_(Tensor &x);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_SQUARE_H__

