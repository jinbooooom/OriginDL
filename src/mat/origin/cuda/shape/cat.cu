#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief cat算子实现（在GPU上直接拼接，避免CPU-GPU传输）
 * @param inputs 输入矩阵列表
 * @param dim 拼接维度
 * @return 拼接结果矩阵
 */
std::unique_ptr<Mat> cat(const std::vector<const origin::OriginMat *> &inputs, int dim)
{
    if (inputs.empty())
    {
        THROW_RUNTIME_ERROR("cat: requires at least 1 input");
    }

    if (inputs.size() == 1)
    {
        // 只有一个输入，直接复制
        return std::make_unique<OriginMat>(*inputs[0]);
    }

    // 检查所有输入的形状和设备
    auto first_shape = inputs[0]->shape();
    Device device    = inputs[0]->device();
    DataType dtype   = inputs[0]->dtype();

    for (size_t i = 1; i < inputs.size(); ++i)
    {
        auto shape = inputs[i]->shape();
        if (shape.size() != first_shape.size())
        {
            THROW_RUNTIME_ERROR("cat: all inputs must have same number of dimensions");
        }

        for (size_t d = 0; d < shape.size(); ++d)
        {
            if (d != static_cast<size_t>(dim) && shape[d] != first_shape[d])
            {
                THROW_RUNTIME_ERROR("cat: dimension {} mismatch: {} vs {}", d, shape[d], first_shape[d]);
            }
        }

        VALIDATE_SAME_DTYPE(*inputs[0], *inputs[i]);
        VALIDATE_SAME_CUDA_DEVICE(*inputs[0], *inputs[i]);
    }

    // 计算输出形状
    Shape output_shape    = first_shape;
    size_t total_dim_size = 0;
    for (const auto *x : inputs)
    {
        total_dim_size += x->shape()[dim];
    }
    output_shape[dim] = total_dim_size;

    // 创建输出矩阵
    auto result = std::make_unique<OriginMat>(output_shape, dtype, device);

    // 计算每个维度的步长（用于计算内存偏移）
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

    auto output_strides       = compute_strides(output_shape);
    size_t element_size_bytes = element_size(dtype);

    // 计算每个输入在输出中的偏移量（以元素为单位）
    size_t output_offset_elements = 0;

    for (const auto *input : inputs)
    {
        auto input_shape = input->shape();

        // 计算输出中的目标偏移（字节）
        // 对于 dim=1 的情况，例如 [B, C1, H, W] cat [B, C2, H, W] -> [B, C1+C2, H, W]
        // 我们需要计算每个"切片"的大小和偏移
        // 切片大小 = H * W（dim 之后的所有维度）
        size_t slice_size = 1;
        for (size_t d = dim + 1; d < output_shape.size(); ++d)
        {
            slice_size *= output_shape[d];
        }

        // 计算每个输入在拼接维度上的大小
        size_t input_dim_size = input_shape[dim];

        // 计算需要复制的切片数量（dim 之前的所有维度）
        size_t num_slices = 1;
        for (size_t d = 0; d < static_cast<size_t>(dim); ++d)
        {
            num_slices *= input_shape[d];
        }

        // 每个切片在输入中的大小（元素数）
        size_t input_slice_elements = input_dim_size * slice_size;
        size_t input_slice_bytes    = input_slice_elements * element_size_bytes;

        // 每个切片在输出中的大小（元素数）
        size_t output_slice_elements = output_shape[dim] * slice_size;
        size_t output_slice_bytes    = output_slice_elements * element_size_bytes;

        // 使用 cudaMemcpy 批量复制每个切片
        const void *src = input->storage()->data();
        void *dst_base  = result->storage()->data();

        for (size_t slice = 0; slice < num_slices; ++slice)
        {
            const void *src_slice = static_cast<const char *>(src) + slice * input_slice_bytes;
            void *dst_slice =
                static_cast<char *>(dst_base) +
                (slice * output_slice_elements + output_offset_elements * slice_size) * element_size_bytes;
            CUDA_CHECK(cudaMemcpy(dst_slice, src_slice, input_slice_bytes, cudaMemcpyDeviceToDevice));
        }

        // 更新输出偏移（以元素为单位）
        output_offset_elements += input_dim_size;
    }

    CUDA_CHECK_ASYNC();
    return result;
}

/**
 * @brief CUDA split：将矩阵沿指定维度分割成多个矩阵（cat 的反向操作）
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

            CUDA_CHECK(cudaMemcpy(output_slice, input_slice, output_slice_bytes, cudaMemcpyDeviceToDevice));
        }

        results.push_back(std::move(result));
        input_offset_elements += output_dim_size;
    }

    CUDA_CHECK_ASYNC();
    return results;
}

}  // namespace cuda
}  // namespace origin
