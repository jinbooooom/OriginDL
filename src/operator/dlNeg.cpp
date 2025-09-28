#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Neg::forward(const std::vector<Tensor> &xs)
{
    // 使用抽象层进行负号运算
    auto result = -mat(xs[0]);
    auto y      = convert_mat_to_tensor(std::move(result));
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

    // 使用抽象层进行梯度计算
    auto result = -mat(gys[0]);
    auto gx     = convert_mat_to_tensor(std::move(result));
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
