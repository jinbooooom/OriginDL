#ifndef __ORIGIN_DL_TRANSPOSE_H__
#define __ORIGIN_DL_TRANSPOSE_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Transpose : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor transpose(const Tensor &x);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_TRANSPOSE_H__

