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
};

extern Tensor square(const Tensor &x);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_SQUARE_H__

