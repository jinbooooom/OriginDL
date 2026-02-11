#include <memory>
#include <vector>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU permute：按照指定顺序重新排列张量的维度
 * @param mat 输入矩阵
 * @param dims 新的维度顺序，例如 {0, 2, 3, 1} 表示将维度 0,1,2,3 重新排列为 0,2,3,1
 * @return 重排后的矩阵
 */
std::unique_ptr<Mat> permute(const OriginMat &mat, const std::vector<int> &dims)
{
    auto input_shape = mat.shape();
    size_t ndim      = input_shape.size();

    if (unlikely(dims.size() != ndim))
    {
        THROW_INVALID_ARG("permute: dims size {} does not match input dimension {}", dims.size(), ndim);
    }

    // 验证 dims 的有效性
    std::vector<bool> used(ndim, false);
    for (int dim : dims)
    {
        if (unlikely(dim < 0 || dim >= static_cast<int>(ndim)))
        {
            THROW_INVALID_ARG("permute: invalid dimension {} (must be in [0, {}))", dim, ndim);
        }
        if (unlikely(used[dim]))
        {
            THROW_INVALID_ARG("permute: duplicate dimension {}", dim);
        }
        used[dim] = true;
    }

    // 计算输出形状
    std::vector<size_t> output_dims;
    output_dims.reserve(ndim);
    for (int dim : dims)
    {
        output_dims.push_back(input_shape[dim]);
    }
    Shape output_shape(output_dims);

    // 创建输出矩阵
    auto result = std::make_unique<OriginMat>(output_shape, mat.dtype(), mat.device());

    // 计算输入和输出的步长
    auto compute_strides = [](const Shape &shape) {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    };

    auto input_strides  = compute_strides(input_shape);
    auto output_strides = compute_strides(output_shape);

    // 使用类型分发器执行 permute 操作
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        const T *input_data = mat.data_ptr<T>();
        T *output_data      = result->data_ptr<T>();

        size_t total_elements = mat.elements();

        // 对每个输出元素，计算对应的输入索引
        for (size_t out_idx = 0; out_idx < total_elements; ++out_idx)
        {
            // 将输出索引转换为坐标
            std::vector<size_t> out_coords(ndim);
            size_t remaining = out_idx;
            for (size_t d = 0; d < ndim; ++d)
            {
                out_coords[d] = remaining / output_strides[d];
                remaining %= output_strides[d];
            }

            // 将输出坐标映射回输入坐标
            std::vector<size_t> in_coords(ndim);
            for (size_t d = 0; d < ndim; ++d)
            {
                in_coords[dims[d]] = out_coords[d];
            }

            // 计算输入索引
            size_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d)
            {
                in_idx += in_coords[d] * input_strides[d];
            }

            output_data[out_idx] = input_data[in_idx];
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
