#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/type_promotion.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> Sub::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        THROW_RUNTIME_ERROR("Sub operator requires exactly 2 inputs, but got {}", xs.size());
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();

    // 检查类型是否匹配，如果不匹配则进行类型提升
    if (xs[0].dtype() != xs[1].dtype())
    {
        // 自动类型提升
        DataType promoted_type = promote_types_rule(xs[0].dtype(), xs[1].dtype());
        Tensor x0              = xs[0].dtype() == promoted_type ? xs[0] : xs[0].to(promoted_type);
        Tensor x1              = xs[1].dtype() == promoted_type ? xs[1] : xs[1].to(promoted_type);

        // 使用提升后的张量进行运算
        auto result = mat(x0) - mat(x1);
        auto y      = convert_mat_to_tensor(std::move(result));

        std::vector<Tensor> outputs;
        outputs.push_back(y);
        return outputs;
    }

    // 类型匹配，直接运算
    auto result = mat(xs[0]) - mat(xs[1]);
    auto y      = convert_mat_to_tensor(std::move(result));
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Sub::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Sub backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // TODO: 未来需要在backward中也实现类型提升逻辑

    auto gx0 = gys[0];
    // 使用负号运算符
    auto gx1_result = -mat(gys[0]);
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

Tensor sub(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Sub>();
    return (*op)(xs)[0];
}

Tensor sub(const Tensor &lhs, const Tensor &rhs)
{
    return sub({lhs, rhs});
}

Tensor operator-(const Tensor &lhs, const Tensor &rhs)
{
    return sub(lhs, rhs);
}

Tensor operator-(const Tensor &lhs, const Scalar &rhs)
{
    auto x = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    return sub(lhs, x);
}

Tensor operator-(const Scalar &lhs, const Tensor &rhs)
{
    auto x = Tensor(lhs, Shape({}), dtype(lhs.dtype()).device(rhs.device()));
    return sub(x, rhs);
}

}  // namespace origin