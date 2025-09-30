#include "base/utils.h"
#include "operator.h"

namespace origin
{

std::vector<Tensor> Sum::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        throw std::runtime_error("Sum requires exactly 1 input");
    }
    // 使用抽象层进行求和运算
    auto result = mat(xs[0]).sum(this->axis_);
    auto y      = convert_mat_to_tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Sum::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("Sum backward requires exactly 1 gradient");
    }
    auto gy = &mat(gys[0]);
    auto x  = &mat(this->inputs_[0]);

    // 使用抽象层进行梯度计算
    auto x_shape   = x->shape();
    auto gx_result = gy->broadcast_to(x_shape);
    auto gx        = convert_mat_to_tensor(std::move(gx_result));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor sum(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Sum>(-1);  // -1 意味着所有元素相加
    return (*op)(xs)[0];
}

Tensor sum(const Tensor &x, int axis)
{
    auto op                    = std::make_shared<Sum>(axis);
    std::vector<Tensor> inputs = {x};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace origin