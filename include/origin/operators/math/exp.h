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
};

extern Tensor exp(const Tensor &x);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_EXP_H__

