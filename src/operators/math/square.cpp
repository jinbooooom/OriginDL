#include "origin/core/operator.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Square::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Square operator requires exactly 1 input, but got {}", xs.size());
    }

    auto result = mat(xs[0]).square();
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Square::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Square backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &x  = mat(this->inputs_[0]);
    auto &gy = mat(gys[0]);

    auto tmp      = x * gy;
    auto scalar_2 = Tensor(2, Shape({}), dtype(this->inputs_[0].dtype()).device(this->inputs_[0].device()));
    tmp->mul_inplace(mat(scalar_2));
    auto gx = convert_mat_to_tensor(std::move(tmp));
    return std::vector<Tensor>{std::move(gx)};
}

void Square::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (unlikely(&input1 != &kNullTensor_))
    {
        THROW_INVALID_ARG("Square is a unary operator, cannot accept two operands");
    }

    mat(input0).square_inplace();
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

void square_(Tensor &x)
{
    // 创建 Square 实例并调用 forward_inplace
    Square op;
    op.forward_inplace(x, Operator::kNullTensor_);
}

}  // namespace functional
}  // namespace origin