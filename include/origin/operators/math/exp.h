#ifndef __ORIGIN_DL_EXP_H__
#define __ORIGIN_DL_EXP_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Exp : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    void forward_inplace(Tensor &input0, const Tensor &input1) override;
};

extern Tensor exp(const Tensor &x);

// 原地操作函数
extern void exp_(Tensor &x);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_EXP_H__

