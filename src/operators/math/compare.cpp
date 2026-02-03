#include "origin/operators/math/compare.h"
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

// ==================== Equal 算子实现 ====================

std::vector<Tensor> Equal::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("Equal operator requires exactly 2 inputs (tensor, scalar), but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto &threshold = xs[1];

    // threshold 可以是标量（shape为{}或{1}）或与x相同形状的张量
    auto result = mat(x) == mat(threshold);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Equal::backward(const std::vector<Tensor> &gys)
{
    // 比较运算符的反向传播：梯度为零（参考 PyTorch 的实现）
    // 比较运算符是不可微的，所以对输入的梯度为零
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Equal backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    // 返回全零梯度：第一个是对 xs[0] 的梯度，第二个是对 xs[1] 的梯度
    auto gx0 = Tensor::zeros(inputs_[0].shape(), dtype(gy.dtype()).device(gy.device()));
    auto gx1 = Tensor::zeros(inputs_[1].shape(), dtype(gy.dtype()).device(gy.device()));
    return std::vector<Tensor>{std::move(gx0), std::move(gx1)};
}

// ==================== NotEqual 算子实现 ====================

std::vector<Tensor> NotEqual::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("NotEqual operator requires exactly 2 inputs (tensor, scalar), but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto &threshold = xs[1];

    auto result = mat(x) != mat(threshold);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> NotEqual::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("NotEqual backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto gx0 = Tensor::zeros(inputs_[0].shape(), dtype(gy.dtype()).device(gy.device()));
    auto gx1 = Tensor::zeros(inputs_[1].shape(), dtype(gy.dtype()).device(gy.device()));
    return std::vector<Tensor>{std::move(gx0), std::move(gx1)};
}

// ==================== Less 算子实现 ====================

std::vector<Tensor> Less::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("Less operator requires exactly 2 inputs (tensor, scalar), but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto &threshold = xs[1];

    auto result = mat(x) < mat(threshold);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Less::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Less backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto gx0 = Tensor::zeros(inputs_[0].shape(), dtype(gy.dtype()).device(gy.device()));
    auto gx1 = Tensor::zeros(inputs_[1].shape(), dtype(gy.dtype()).device(gy.device()));
    return std::vector<Tensor>{std::move(gx0), std::move(gx1)};
}

// ==================== LessEqual 算子实现 ====================

std::vector<Tensor> LessEqual::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("LessEqual operator requires exactly 2 inputs (tensor, scalar), but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto &threshold = xs[1];

    auto result = mat(x) <= mat(threshold);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> LessEqual::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("LessEqual backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto gx0 = Tensor::zeros(inputs_[0].shape(), dtype(gy.dtype()).device(gy.device()));
    auto gx1 = Tensor::zeros(inputs_[1].shape(), dtype(gy.dtype()).device(gy.device()));
    return std::vector<Tensor>{std::move(gx0), std::move(gx1)};
}

// ==================== Greater 算子实现 ====================

std::vector<Tensor> Greater::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("Greater operator requires exactly 2 inputs (tensor, scalar), but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto &threshold = xs[1];

    auto result = mat(x) > mat(threshold);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Greater::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Greater backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto gx0 = Tensor::zeros(inputs_[0].shape(), dtype(gy.dtype()).device(gy.device()));
    auto gx1 = Tensor::zeros(inputs_[1].shape(), dtype(gy.dtype()).device(gy.device()));
    return std::vector<Tensor>{std::move(gx0), std::move(gx1)};
}

// ==================== GreaterEqual 算子实现 ====================

std::vector<Tensor> GreaterEqual::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("GreaterEqual operator requires exactly 2 inputs (tensor, scalar), but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto &threshold = xs[1];

    auto result = mat(x) >= mat(threshold);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> GreaterEqual::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("GreaterEqual backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto gx0 = Tensor::zeros(inputs_[0].shape(), dtype(gy.dtype()).device(gy.device()));
    auto gx1 = Tensor::zeros(inputs_[1].shape(), dtype(gy.dtype()).device(gy.device()));
    return std::vector<Tensor>{std::move(gx0), std::move(gx1)};
}

}  // namespace functional

// ==================== 全局运算符重载 ====================

// 对张量的比较运算符
Tensor operator==(const Tensor &lhs, const Tensor &rhs)
{
    auto op = std::make_shared<functional::Equal>();
    return (*op)({lhs, rhs})[0];
}

Tensor operator!=(const Tensor &lhs, const Tensor &rhs)
{
    auto op = std::make_shared<functional::NotEqual>();
    return (*op)({lhs, rhs})[0];
}

Tensor operator<(const Tensor &lhs, const Tensor &rhs)
{
    auto op = std::make_shared<functional::Less>();
    return (*op)({lhs, rhs})[0];
}

Tensor operator<=(const Tensor &lhs, const Tensor &rhs)
{
    auto op = std::make_shared<functional::LessEqual>();
    return (*op)({lhs, rhs})[0];
}

Tensor operator>(const Tensor &lhs, const Tensor &rhs)
{
    auto op = std::make_shared<functional::Greater>();
    return (*op)({lhs, rhs})[0];
}

Tensor operator>=(const Tensor &lhs, const Tensor &rhs)
{
    auto op = std::make_shared<functional::GreaterEqual>();
    return (*op)({lhs, rhs})[0];
}

// 对标量的比较运算符
Tensor operator==(const Tensor &lhs, const Scalar &rhs)
{
    auto scalar_tensor = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    auto op            = std::make_shared<functional::Equal>();
    return (*op)({lhs, scalar_tensor})[0];
}

Tensor operator!=(const Tensor &lhs, const Scalar &rhs)
{
    auto scalar_tensor = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    auto op            = std::make_shared<functional::NotEqual>();
    return (*op)({lhs, scalar_tensor})[0];
}

Tensor operator<(const Tensor &lhs, const Scalar &rhs)
{
    auto scalar_tensor = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    auto op            = std::make_shared<functional::Less>();
    return (*op)({lhs, scalar_tensor})[0];
}

Tensor operator<=(const Tensor &lhs, const Scalar &rhs)
{
    auto scalar_tensor = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    auto op            = std::make_shared<functional::LessEqual>();
    return (*op)({lhs, scalar_tensor})[0];
}

Tensor operator>(const Tensor &lhs, const Scalar &rhs)
{
    auto scalar_tensor = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    auto op            = std::make_shared<functional::Greater>();
    return (*op)({lhs, scalar_tensor})[0];
}

Tensor operator>=(const Tensor &lhs, const Scalar &rhs)
{
    auto scalar_tensor = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    auto op            = std::make_shared<functional::GreaterEqual>();
    return (*op)({lhs, scalar_tensor})[0];
}

}  // namespace origin
