#ifndef __ORIGIN_DL_MAT_MUL_H__
#define __ORIGIN_DL_MAT_MUL_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class MatMul : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor mat_mul(const Tensor &x, const Tensor &w);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_MAT_MUL_H__
