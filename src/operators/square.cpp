#include "origin/core/operator.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> Square::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Square operator requires exactly 1 input, but got {}", xs.size());
    }
    // 使用抽象层进行平方运算
    auto result = mat(xs[0]) * mat(xs[0]);
    auto y      = convert_mat_to_tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Square::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Square backward requires exactly 1 gradient, but got {}", gys.size());
    }
    auto &x  = mat(this->inputs_[0]);
    auto &gy = mat(gys[0]);

    // 使用抽象层进行梯度计算
    auto temp_mult = x * gy;
    // 创建标量2.0的张量
    auto scalar_2  = Tensor(2, Shape({}), dtype(x.dtype()).device(x.device()));
    auto gx_result = *temp_mult * mat(scalar_2);
    auto gx        = convert_mat_to_tensor(std::move(gx_result));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor square(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Square>();
    return (*op)(xs)[0];
}

Tensor square(const Tensor &x)
{
    auto xs = std::vector<Tensor>();
    xs.emplace_back(x);
    return square(xs);
}

}  // namespace origin