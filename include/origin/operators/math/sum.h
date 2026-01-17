#ifndef __ORIGIN_DL_SUM_H__
#define __ORIGIN_DL_SUM_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class Sum : public Operator
{
public:
    int axis_;  // 对那个轴求和

    Shape x_shape_;  // 输入的形状
    Sum() : axis_(-1){};
    Sum(const int axis) : axis_(axis){};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor sum(const Tensor &x, int axis = -1);  // -1 意味着所有元素相加

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_SUM_H__
