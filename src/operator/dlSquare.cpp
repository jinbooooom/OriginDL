#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Square::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1) {
        throw std::runtime_error("Square requires exactly 1 input");
    }
    auto x = xs[0].data();
    // 直接使用 ArrayFire 的运算符，避免触发全局 operator^
    auto y = Tensor(x * x);
    return std::vector<Tensor>{y};
}

std::vector<Tensor> Square::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("Square backward requires exactly 1 gradient");
    }
    auto x = this->inputs_[0].data();
    auto gy = gys[0].data();
    auto gx = Tensor(2.0 * x * gy);
    return std::vector<Tensor>{gx};
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

}  // namespace dl