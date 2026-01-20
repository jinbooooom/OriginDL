#include "origin/core/operator.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Log::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Log operator requires exactly 1 input, but got {}", xs.size());
    }

    // 使用抽象层进行自然对数运算
    auto result = mat(xs[0]).log();
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Log::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Log backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // ln(x) 的梯度：∂y/∂x = 1/x
    // 所以 gx = gy / x
    auto &x  = mat(this->inputs_[0]);
    auto &gy = mat(gys[0]);

    auto gx_result = gy / x;
    auto gx        = convert_mat_to_tensor(std::move(gx_result));
    return std::vector<Tensor>{std::move(gx)};
}

void Log::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (unlikely(&input1 != &kNullTensor_))
    {
        THROW_INVALID_ARG("Log is a unary operator, cannot accept two operands");
    }

    // 原地操作：input0 = log(input0)
    mat(input0).log_inplace();
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

void log_(Tensor &x)
{
    // 创建 Log 实例并调用 forward_inplace
    Log op;
    op.forward_inplace(x, Operator::kNullTensor_);
}

}  // namespace functional
}  // namespace origin
