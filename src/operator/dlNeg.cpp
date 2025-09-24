#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Neg::forward(const std::vector<Tensor> &xs)
{
    // 直接使用 ArrayFire 运算符，避免递归调用
    auto y = Tensor(-xs[0].data());
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Neg::backward(const std::vector<Tensor> &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    // 直接使用 ArrayFire 运算符，避免递归调用
    auto gx = Tensor(-gys[0].data());
    std::vector<Tensor> gxs;
    gxs.push_back(gx);

    return gxs;
}

Tensor neg(const std::vector<Tensor> &xs)
{
    return (*std::shared_ptr<Operator>(new Neg()))(xs)[0];
}

Tensor neg(const Tensor &x)
{
    auto xs = std::vector<Tensor>();
    xs.emplace_back(x);
    return neg(xs);
}

Tensor operator-(const Tensor &x)
{
    return neg(x);
}

}  // namespace dl
