#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> Add::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        THROW_RUNTIME_ERROR("Add operator requires exactly 2 inputs, but got {}", xs.size());
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();

    // 检查类型是否匹配，如果不匹配则进行类型提升
    if (xs[0].dtype() != xs[1].dtype())
    {
        // 自动类型提升
        DataType promoted_type = promote_types(xs[0].dtype(), xs[1].dtype());
        Tensor x0              = xs[0].dtype() == promoted_type ? xs[0] : xs[0].to(promoted_type);
        Tensor x1              = xs[1].dtype() == promoted_type ? xs[1] : xs[1].to(promoted_type);

        // 使用提升后的张量进行运算
        auto result = mat(x0) + mat(x1);
        auto y      = convert_mat_to_tensor(std::move(result));

        std::vector<Tensor> outputs;
        outputs.push_back(y);
        return outputs;
    } 

    // 类型匹配，直接运算
    auto result = mat(xs[0]) + mat(xs[1]);
    auto y      = convert_mat_to_tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Add::backward(const std::vector<Tensor> &gys)
{
    if (1 != gys.size())
    {
        THROW_RUNTIME_ERROR("Add backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto gx0 = gys[0];
    auto gx1 = gys[0];
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

Tensor operator+(const Tensor &lhs, const Scalar &rhs)
{
    auto x     = create_tensor_from_scalar(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    return add(lhs, x);
}

Tensor operator+(const Scalar &lhs, const Tensor &rhs)
{
    return rhs + lhs;
}

// 模板版本的operator+函数
template <typename T>
Tensor operator+(const Tensor &lhs, T rhs)
{
    auto x = Tensor(rhs, Shape({}), dtype(get_data_type_from_template<T>()).device(lhs.device()));
    return add(lhs, x);
}

template <typename T>
Tensor operator+(T lhs, const Tensor &rhs)
{
    return rhs + lhs;
}

// 显式实例化
template Tensor operator+<float>(const Tensor &lhs, float rhs);
template Tensor operator+<float>(float lhs, const Tensor &rhs);
template Tensor operator+<double>(const Tensor &lhs, double rhs);
template Tensor operator+<double>(double lhs, const Tensor &rhs);
template Tensor operator+<int32_t>(const Tensor &lhs, int32_t rhs);
template Tensor operator+<int32_t>(int32_t lhs, const Tensor &rhs);

}  // namespace origin
