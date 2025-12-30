#include "origin/core/operator.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> Log::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Log operator requires exactly 1 input, but got {}", xs.size());
    }

    // 使用抽象层进行自然对数运算
    auto result = mat(xs[0]).log();
    auto y      = convert_mat_to_tensor(std::move(result));
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Log::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Log backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // ln(x) 的梯度：∂y/∂x = 1/x
    // 所以 gx = gy / x
    auto &x        = mat(this->inputs_[0]);
    auto &gy       = mat(gys[0]);
    
    auto gx_result = gy / x;
    auto gx        = convert_mat_to_tensor(std::move(gx_result));
    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor log(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Log>();
    return (*op)(xs)[0];
}

Tensor log(const Tensor &x)
{
    auto xs = std::vector<Tensor>();
    xs.emplace_back(x);
    return log(xs);
}

}  // namespace origin

