#include "origin/core/operator.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Sigmoid::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Sigmoid operator requires exactly 1 input, but got {}", xs.size());
    }

    // Sigmoid: y = 1 / (1 + exp(-x))
    auto &x = xs[0];

    // 计算 -x
    auto neg_x_result = -mat(x);
    auto neg_x        = convert_mat_to_tensor(std::move(neg_x_result));

    // 计算 exp(-x)
    auto exp_neg_x = exp(neg_x);

    // 创建全1张量，形状与exp_neg_x相同
    auto ones = Tensor::ones(exp_neg_x.shape(), dtype(exp_neg_x.dtype()).device(exp_neg_x.device()));

    // 计算 1 + exp(-x)
    auto one_plus_exp = ones + exp_neg_x;

    // 计算 1 / (1 + exp(-x))
    auto result = ones / one_plus_exp;

    // 保存 sigmoid(x) 用于反向传播
    sigmoid_x_ = result;

    return std::vector<Tensor>{std::move(result)};
}

std::vector<Tensor> Sigmoid::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Sigmoid backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // Sigmoid 的梯度：gx = gy * sigmoid(x) * (1 - sigmoid(x))
    // 直接使用 forward 中保存的 sigmoid_x_
    auto &gy = gys[0];

    // 计算 1 - sigmoid_x_
    auto ones              = Tensor::ones(sigmoid_x_.shape(), dtype(sigmoid_x_.dtype()).device(sigmoid_x_.device()));
    auto one_minus_sigmoid = ones - sigmoid_x_;

    // 计算 gx = gy * sigmoid_x_ * (1 - sigmoid_x_)
    auto temp = gy * sigmoid_x_;
    auto gx   = temp * one_minus_sigmoid;
    return std::vector<Tensor>{std::move(gx)};
}

Tensor sigmoid(const Tensor &x)
{
    auto op = std::make_shared<Sigmoid>();
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
