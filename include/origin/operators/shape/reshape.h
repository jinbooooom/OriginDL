#ifndef __ORIGIN_DL_RESHAPE_H__
#define __ORIGIN_DL_RESHAPE_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Reshape : public Operator
{
public:
    Shape shape_;  // 输出的形状

    Shape x_shape_;  // 输入的形状

    Reshape(const Shape &shape) : shape_(shape) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor reshape(const Tensor &x, const Shape &shape);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_RESHAPE_H__

