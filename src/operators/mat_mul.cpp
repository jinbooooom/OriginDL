#include "origin/core/operator.h"

namespace origin
{

std::vector<Tensor> MatMul::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        throw std::runtime_error("MatMul requires exactly 2 inputs");
    }

    // 使用抽象层进行矩阵乘法运算
    auto result = mat(xs[0]).matmul(mat(xs[1]));
    auto y      = convert_mat_to_tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> MatMul::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("MatMul backward requires exactly 1 gradient");
    }

    auto x  = &mat(this->inputs_[0]);
    auto w  = &mat(this->inputs_[1]);
    auto gy = &mat(gys[0]);

    // 使用抽象层进行梯度计算
    auto w_T = w->transpose();
    auto x_T = x->transpose();

    auto gx_result = gy->matmul(*w_T);
    auto gw_result = x_T->matmul(*gy);

    auto gx = convert_mat_to_tensor(std::move(gx_result));
    auto gw = convert_mat_to_tensor(std::move(gw_result));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    outputs.push_back(gw);
    return outputs;
}

Tensor matmul(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<MatMul>();
    return (*op)(xs)[0];
}

Tensor matmul(const Tensor &lhs, const Tensor &rhs)
{
    return matmul({lhs, rhs});
}

Tensor mat_mul(const Tensor &lhs, const Tensor &rhs)
{
    return matmul(lhs, rhs);
}

}  // namespace origin