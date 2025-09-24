#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Mul::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2) {
        throw std::runtime_error("Mul requires exactly 2 inputs");
    }
    
    shape0_ = xs[0].data().dims();
    shape1_ = xs[1].data().dims();
    // 直接使用 ArrayFire 运算符，避免触发全局 operator*
    auto y = Tensor(xs[0].data() * xs[1].data());
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Mul::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("Mul backward requires exactly 1 gradient");
    }

    auto x0 = this->inputs_[0].data();
    auto x1 = this->inputs_[1].data();
    auto gy = gys[0].data();
    
    // 直接使用 ArrayFire 的运算符，避免触发全局 operator*
    auto gx0 = Tensor(gy * x1);
    auto gx1 = Tensor(gy * x0);
    
    if (shape0_ != shape1_) {
        // 这里需要实现 sum_to 功能
        // gx0 = sum_to(gx0, shape0_);
        // gx1 = sum_to(gx1, shape1_);
    }
    
    std::vector<Tensor> gxs;
    gxs.push_back(gx0);
    gxs.push_back(gx1);
    return gxs;
}

Tensor mul(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Mul>();
    return (*op)(xs)[0];
}

Tensor mul(const Tensor &lhs, const Tensor &rhs)
{
    return mul({lhs, rhs});
}

Tensor operator*(const Tensor &lhs, const Tensor &rhs)
{
    return mul(lhs, rhs);
}

Tensor operator*(const Tensor &lhs, data_t rhs)
{
    auto dims = lhs.data().dims();
    auto x = Tensor(af::constant(rhs, dims));
    return mul(lhs, x);
}

Tensor operator*(data_t lhs, const Tensor &rhs)
{
    auto dims = rhs.data().dims();
    auto x = Tensor(af::constant(lhs, dims));
    return mul(x, rhs);
}

}  // namespace dl