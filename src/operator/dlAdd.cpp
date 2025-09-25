#include "base/dlException.h"
#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Add::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2) {
        throw std::runtime_error("Add requires exactly 2 inputs");
    }
    
    shape0_ = xs[0].data().dims();
    shape1_ = xs[1].data().dims();
    // 直接使用 ArrayFire 运算符，避免触发全局 operator+
    auto y = Tensor(xs[0].data() + xs[1].data());
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Add::backward(const std::vector<Tensor> &gys)
{
    if (1 != gys.size())
    {
        DL_WARN_THROW("invalid argument size, not equal to 1");
    }

    auto gx0 = gys[0];
    auto gx1 = gys[0];
    if (shape0_ != shape1_)
    {
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

Tensor add(const std::vector<Tensor> &xs)
{
    return (*std::shared_ptr<Operator>(new Add()))(xs)[0];
}

Tensor add(const Tensor &lhs, const Tensor &rhs)
{
    return add({lhs, rhs});
}

Tensor operator+(const Tensor &lhs, const Tensor &rhs)
{
    return add(lhs, rhs);
}

Tensor operator+(const Tensor &lhs, data_t rhs)
{
    auto dims = lhs.data().dims();
    auto x = Tensor(af::constant(rhs, dims));
    return add(lhs, x);
}

Tensor operator+(data_t lhs, const Tensor &rhs)
{
    auto dims = rhs.data().dims();
    auto x = Tensor(af::constant(lhs, dims));
    return add(x, rhs);
}

}  // namespace dl
