#include <cstring>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU split：将矩阵沿指定维度分割成多个矩阵（cat 的反向操作）
 * @param input 输入矩阵
 * @param output_shapes 输出形状列表
 * @param dim 分割维度
 * @return 分割后的矩阵列表
 */
std::vector<std::unique_ptr<Mat>> split(const OriginMat &input, const std::vector<Shape> &output_shapes, int dim)
{
    if (output_shapes.empty())
    {
        THROW_RUNTIME_ERROR("split: requires at least 1 output shape");
    }

    auto input_shape          = input.shape();
    size_t element_size_bytes = element_size(input.dtype());

    // 验证输出形状的总和等于输入形状
    size_t total_dim_size = 0;
    for (const auto &shape : output_shapes)
    {
        if (shape.size() != input_shape.size())
        {
            THROW_RUNTIME_ERROR("split: output shape dimension mismatch. Expected {}, got {}", input_shape.size(),
                                shape.size());
        }
        for (size_t d = 0; d < shape.size(); ++d)
        {
            if (d != static_cast<size_t>(dim) && shape[d] != input_shape[d])
            {
                THROW_RUNTIME_ERROR("split: dimension {} mismatch: {} vs {}", d, shape[d], input_shape[d]);
            }
        }
        total_dim_size += shape[dim];
    }
    if (total_dim_size != input_shape[dim])
    {
        THROW_RUNTIME_ERROR("split: total output dimension size {} does not match input dimension size {}",
                            total_dim_size, input_shape[dim]);
    }

    std::vector<std::unique_ptr<Mat>> results;
    results.reserve(output_shapes.size());

    // 计算切片大小（dim 之后的所有维度）
    size_t slice_size = 1;
    for (size_t d = dim + 1; d < input_shape.size(); ++d)
    {
        slice_size *= input_shape[d];
    }

    // 计算需要复制的切片数量（dim 之前的所有维度）
    size_t num_slices = 1;
    for (size_t d = 0; d < static_cast<size_t>(dim); ++d)
    {
        num_slices *= input_shape[d];
    }

    const void *input_data       = input.storage()->data();
    size_t input_offset_elements = 0;

    for (const auto &output_shape : output_shapes)
    {
        // 创建输出矩阵
        auto result       = std::make_unique<OriginMat>(output_shape, input.dtype(), input.device());
        void *output_data = result->storage()->data();

        size_t output_dim_size       = output_shape[dim];
        size_t output_slice_elements = output_dim_size * slice_size;
        size_t output_slice_bytes    = output_slice_elements * element_size_bytes;

        // 复制每个切片
        for (size_t slice = 0; slice < num_slices; ++slice)
        {
            size_t input_slice_elements = input_shape[dim] * slice_size;
            const void *input_slice =
                static_cast<const char *>(input_data) +
                (slice * input_slice_elements + input_offset_elements * slice_size) * element_size_bytes;
            void *output_slice = static_cast<char *>(output_data) + slice * output_slice_bytes;

            std::memcpy(output_slice, input_slice, output_slice_bytes);
        }

        results.push_back(std::move(result));
        input_offset_elements += output_dim_size;
    }

    return results;
}

}  // namespace cpu
}  // namespace origin
