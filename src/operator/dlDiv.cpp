#include "base/dlException.h"
#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Div::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        throw std::runtime_error("Div requires exactly 2 inputs");
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();
    // 使用抽象层进行除法运算
    auto result = mat(xs[0]) / mat(xs[1]);
    auto y      = convert_mat_to_tensor(std::move(result));
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Div::backward(const std::vector<Tensor> &gys)
{
    if (1 != gys.size())
    {
        DL_WARN_THROW("invalid argument size, not equal to 1");
    }

    // 正确的除法导数计算：
    // 对于 y = x0 / x1：
    // ∂y/∂x0 = 1/x1
    // ∂y/∂x1 = -x0/x1²
    auto x0 = &mat(this->inputs_[0]);
    auto x1 = &mat(this->inputs_[1]);
    auto gy = &mat(gys[0]);

    // ∂y/∂x0 = gy * (1/x1) = gy / x1
    auto gx0_result = *gy / *x1;
    auto gx0        = convert_mat_to_tensor(std::move(gx0_result));

    // ∂y/∂x1 = gy * (-x0/x1²) = -gy * x0 / x1²
    auto x1_squared = *x1 * *x1;
    auto temp_mult  = *gy * *x0;
    auto temp_div   = *temp_mult / *x1_squared;
    auto gx1_result = -*temp_div;
    auto gx1        = convert_mat_to_tensor(std::move(gx1_result));

    if (shape0_ != shape1_)
    {
        // 实现 sum_to 功能：将梯度广播回原始形状
        if (gx0.shape() != shape0_)
        {
            gx0 = sum_to(gx0, shape0_);
        }
        if (gx1.shape() != shape1_)
        {
            gx1 = sum_to(gx1, shape1_);
        }
    }

    std::vector<Tensor> gxs;
    gxs.push_back(gx0);
    gxs.push_back(gx1);
    return gxs;
}

Tensor div(const std::vector<Tensor> &xs)
{
    return (*std::shared_ptr<Operator>(new Div()))(xs)[0];
}

Tensor div(const Tensor &lhs, const Tensor &rhs)
{
    return div({lhs, rhs});
}

Tensor operator/(const Tensor &lhs, const Tensor &rhs)
{
    return div(lhs, rhs);
}

Tensor operator/(const Tensor &lhs, data_t rhs)
{
    auto dims = lhs.shape();
    auto x    = Tensor::constant(rhs, dims);
    return div(lhs, x);
}

Tensor operator/(data_t lhs, const Tensor &rhs)
{
    auto dims = rhs.shape();
    auto x    = Tensor::constant(lhs, dims);
    return div(x, rhs);
}

}  // namespace dl
