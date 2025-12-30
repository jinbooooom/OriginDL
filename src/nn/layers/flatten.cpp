#include "origin/nn/layers/flatten.h"
#include <vector>
#include "origin/core/operator.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace nn
{

Flatten::Flatten(int start_dim, int end_dim) : start_dim_(start_dim), end_dim_(end_dim)
{
    if (start_dim < 0)
    {
        THROW_INVALID_ARG("Flatten: start_dim must be non-negative, but got {}", start_dim);
    }
}

Tensor Flatten::forward(const Tensor &input)
{
    auto input_shape = input.shape();
    int ndim         = static_cast<int>(input_shape.size());

    // 处理负索引
    int actual_start_dim = start_dim_;
    int actual_end_dim   = (end_dim_ < 0) ? (ndim + end_dim_) : end_dim_;

    if (actual_start_dim < 0 || actual_start_dim >= ndim)
    {
        THROW_INVALID_ARG("Flatten: start_dim {} is out of range for input with {} dimensions", actual_start_dim, ndim);
    }
    if (actual_end_dim < 0 || actual_end_dim >= ndim)
    {
        THROW_INVALID_ARG("Flatten: end_dim {} is out of range for input with {} dimensions", actual_end_dim, ndim);
    }
    if (actual_start_dim > actual_end_dim)
    {
        THROW_INVALID_ARG("Flatten: start_dim {} must be <= end_dim {}", actual_start_dim, actual_end_dim);
    }

    // 计算展平后的形状
    size_t flattened_size = 1;
    for (int i = actual_start_dim; i <= actual_end_dim; ++i)
    {
        flattened_size *= input_shape[i];
    }

    // 构建新的形状
    std::vector<size_t> new_shape_dims;
    // 保留 start_dim 之前的维度
    for (int i = 0; i < actual_start_dim; ++i)
    {
        new_shape_dims.push_back(input_shape[i]);
    }
    // 添加展平后的维度
    new_shape_dims.push_back(flattened_size);
    // 保留 end_dim 之后的维度
    for (int i = actual_end_dim + 1; i < ndim; ++i)
    {
        new_shape_dims.push_back(input_shape[i]);
    }

    Shape new_shape(new_shape_dims);
    return reshape(input, new_shape);
}

}  // namespace nn
}  // namespace origin
