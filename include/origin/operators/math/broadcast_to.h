#ifndef __ORIGIN_DL_BROADCAST_TO_H__
#define __ORIGIN_DL_BROADCAST_TO_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class BroadcastTo : public Operator
{
public:
    Shape shape_;  // 输出的形状

    Shape x_shape_;  // 输入的形状

    BroadcastTo(const Shape &shape) : shape_(shape){};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor broadcast_to(const Tensor &x, const Shape &shape);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_BROADCAST_TO_H__

