#include "origin/operators/shape/flatten.h"
#include "origin/core/tensor.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> FlattenOp::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Flatten operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto x_shape = x.shape();
    auto x_dims  = x_shape.dims();

    // 处理负索引
    int start_dim = start_dim_;
    int end_dim   = end_dim_;

    if (start_dim < 0)
    {
        start_dim = static_cast<int>(x_dims.size()) + start_dim;
    }
    if (end_dim < 0)
    {
        end_dim = static_cast<int>(x_dims.size()) + end_dim;
    }

    if (unlikely(start_dim < 0 || start_dim >= static_cast<int>(x_dims.size()) || end_dim < 0 ||
                 end_dim >= static_cast<int>(x_dims.size()) || start_dim > end_dim))
    {
        THROW_INVALID_ARG("Flatten: invalid dims - start_dim={}, end_dim={}, shape={}", start_dim_, end_dim_,
                          x_shape.to_string());
    }

    // 计算展平后的形状
    std::vector<size_t> output_dims;

    // 保留 start_dim 之前的维度
    for (int i = 0; i < start_dim; ++i)
    {
        output_dims.push_back(x_dims[i]);
    }

    // 展平 start_dim 到 end_dim 的维度
    size_t flattened_size = 1;
    for (int i = start_dim; i <= end_dim; ++i)
    {
        flattened_size *= x_dims[i];
    }
    output_dims.push_back(flattened_size);

    // 保留 end_dim 之后的维度
    for (int i = end_dim + 1; i < static_cast<int>(x_dims.size()); ++i)
    {
        output_dims.push_back(x_dims[i]);
    }

    Shape output_shape(output_dims);

    // 使用 reshape，这会自动处理 GPU
    auto y = functional::reshape(x, output_shape);
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> FlattenOp::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Flatten backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x  = this->inputs_[0];

    // 反向传播就是 reshape 回原始形状
    auto gx = functional::reshape(gy, x.shape());
    return std::vector<Tensor>{std::move(gx)};
}

Tensor flatten(const Tensor &x, int start_dim, int end_dim)
{
    auto op = std::make_shared<FlattenOp>(start_dim, end_dim);
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
