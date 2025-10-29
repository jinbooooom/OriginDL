#include "origin/mat/origin/cuda/cuda_broadcast.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief 计算广播形状实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 广播后的结果形状
 */
Shape compute_broadcast_shape(const OriginMat &a, const OriginMat &b) 
{
    const auto &shape_a = a.shape();
    const auto &shape_b = b.shape();

    // 如果形状相同，直接返回
    if (shape_a == shape_b)
    {
        return shape_a;
    }

    // 如果一个是标量，返回另一个的形状
    if (a.elements() == 1)
    {
        return shape_b;
    }
    if (b.elements() == 1)
    {
        return shape_a;
    }

    // 复杂广播：计算输出形状
    size_t max_dims = std::max(shape_a.size(), shape_b.size());
    std::vector<size_t> result_dims(max_dims);

    for (size_t i = 0; i < max_dims; ++i)
    {
        size_t dim_a = (i < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
        size_t dim_b = (i < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a == 1 || dim_b == 1 || dim_a == dim_b)
        {
            result_dims[max_dims - 1 - i] = std::max(dim_a, dim_b);
        }
        else
        {
            THROW_INVALID_ARG("Cannot broadcast shapes {} and {}", shape_a.to_string(), shape_b.to_string());
        }
    }

    return Shape(result_dims);
}

}  // namespace cuda
}  // namespace origin
