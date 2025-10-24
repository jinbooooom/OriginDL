#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"
#include "origin/mat/type_promotion.h"

namespace origin
{

std::vector<Tensor> MatMul::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        THROW_RUNTIME_ERROR("MatMul operator requires exactly 2 inputs, but got {}", xs.size());
    }

    // 检查类型是否匹配，如果不匹配则进行类型提升
    if (xs[0].dtype() != xs[1].dtype())
    {
        // 自动类型提升
        DataType promoted_type = promote_types_rule(xs[0].dtype(), xs[1].dtype());
        Tensor x0              = xs[0].dtype() == promoted_type ? xs[0] : xs[0].to(promoted_type);
        Tensor x1              = xs[1].dtype() == promoted_type ? xs[1] : xs[1].to(promoted_type);

        // 使用提升后的张量进行运算
        auto result = mat(x0).matmul(mat(x1));
        auto y      = convert_mat_to_tensor(std::move(result));

        std::vector<Tensor> outputs;
        outputs.push_back(y);
        return outputs;
    }

    // 类型匹配，直接运算
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
        THROW_RUNTIME_ERROR("MatMul backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // TODO: 未来需要在backward中也实现类型提升逻辑

    auto &x  = mat(this->inputs_[0]);
    auto &w  = mat(this->inputs_[1]);
    auto &gy = mat(gys[0]);

    // 使用抽象层进行梯度计算
    auto w_T = w.transpose();
    auto x_T = x.transpose();

    auto gx_result = gy.matmul(*w_T);
    auto gw_result = x_T->matmul(gy);

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