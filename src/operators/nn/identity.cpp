#include "origin/operators/nn/identity.h"
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> Identity::forward(const std::vector<Tensor> &xs)
{
    if (xs.empty())
    {
        THROW_RUNTIME_ERROR("Identity operator requires at least 1 input, but got 0");
    }
    
    // Identity 算子：直接返回输入
    // 对于 Detect 层，如果有多个输入，返回最后一个（通常是检测结果）
    // 但为了兼容性，如果有多个输入，返回所有输入
    if (xs.size() == 1)
    {
        return std::vector<Tensor>{xs[0]};
    }
    else
    {
        // 多个输入时，返回最后一个（Detect 层的输出）
        return std::vector<Tensor>{xs.back()};
    }
}

std::vector<Tensor> Identity::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Identity backward requires exactly 1 gradient, but got {}", gys.size());
    }
    
    auto &gy = gys[0];
    auto &x = this->inputs_[0];
    
    return std::vector<Tensor>{gy};
}

Tensor identity(const Tensor &x)
{
    auto op = std::make_shared<Identity>();
    return (*op)(x);
}

}  // namespace origin

