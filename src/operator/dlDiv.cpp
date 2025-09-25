#include "base/dlException.h"
#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Div::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2) {
        throw std::runtime_error("Div requires exactly 2 inputs");
    }
    
    shape0_ = xs[0].data().dims();
    shape1_ = xs[1].data().dims();
    // 直接使用 ArrayFire 运算符，避免触发全局 operator/
    auto y = Tensor(xs[0].data() / xs[1].data());
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
    auto x0 = this->inputs_[0].data();
    auto x1 = this->inputs_[1].data();
    auto gy = gys[0].data();
    
    // ∂y/∂x0 = gy * (1/x1) = gy / x1
    auto gx0 = Tensor(gy / x1);
    
    // ∂y/∂x1 = gy * (-x0/x1²) = -gy * x0 / x1²  
    auto gx1 = Tensor(-gy * x0 / (x1 * x1));
    
    if (shape0_ != shape1_) {
        // 实现 sum_to 功能：将梯度广播回原始形状
        if (gx0.data().dims() != shape0_) {
            gx0 = sum_to(gx0, shape0_);
        }
        if (gx1.data().dims() != shape1_) {
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
    auto dims = lhs.data().dims();
    auto x = Tensor(af::constant(rhs, dims));
    return div(lhs, x);
}

Tensor operator/(data_t lhs, const Tensor &rhs)
{
    auto dims = rhs.data().dims();
    auto x = Tensor(af::constant(lhs, dims));
    return div(x, rhs);
}

}  // namespace dl
