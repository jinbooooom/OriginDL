#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA gather kernel（int32 索引）
 * @tparam T 数据类型
 */
template <typename T>
__global__ void gather_kernel_int32(const T *__restrict__ input,
                                     const int32_t *__restrict__ indices,
                                     T *__restrict__ output,
                                     size_t N,
                                     size_t C)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int32_t idx = indices[i];
        if (idx >= 0 && idx < static_cast<int32_t>(C))
        {
            output[i] = input[i * C + idx];
        }
    }
}

/**
 * @brief CUDA gather kernel（int64 索引）
 * @tparam T 数据类型
 */
template <typename T>
__global__ void gather_kernel_int64(const T *__restrict__ input,
                                     const int64_t *__restrict__ indices,
                                     T *__restrict__ output,
                                     size_t N,
                                     size_t C)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int64_t idx = indices[i];
        if (idx >= 0 && idx < static_cast<int64_t>(C))
        {
            output[i] = input[i * C + idx];
        }
    }
}

/**
 * @brief CUDA one_hot kernel（int32 索引）
 */
__global__ void one_hot_kernel_int32(const int32_t *__restrict__ indices,
                                      float *__restrict__ output,
                                      size_t N,
                                      int num_classes)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int32_t idx = indices[i];
        // 初始化当前行为0
        for (int j = 0; j < num_classes; ++j)
        {
            output[i * num_classes + j] = 0.0f;
        }
        // 设置对应位置为1
        if (idx >= 0 && idx < num_classes)
        {
            output[i * num_classes + idx] = 1.0f;
        }
    }
}

/**
 * @brief CUDA one_hot kernel（int64 索引）
 */
__global__ void one_hot_kernel_int64(const int64_t *__restrict__ indices,
                                      float *__restrict__ output,
                                      size_t N,
                                      int num_classes)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int64_t idx = indices[i];
        // 初始化当前行为0
        for (int j = 0; j < num_classes; ++j)
        {
            output[i * num_classes + j] = 0.0f;
        }
        // 设置对应位置为1
        if (idx >= 0 && idx < num_classes)
        {
            output[i * num_classes + idx] = 1.0f;
        }
    }
}

/**
 * @brief CUDA gather：根据索引从矩阵中提取值
 * @param input 输入矩阵 (N, C)
 * @param indices 索引向量 (N,)，每个元素在 [0, C) 范围内
 * @return 提取的值 (N,)
 */
std::unique_ptr<Mat> gather(const OriginMat &input, const OriginMat &indices)
{
    if (unlikely(input.shape().size() != 2))
    {
        THROW_INVALID_ARG("gather: input must be 2D (N, C), but got shape {}", input.shape().to_string());
    }

    if (unlikely(indices.shape().size() != 1))
    {
        THROW_INVALID_ARG("gather: indices must be 1D (N,), but got shape {}", indices.shape().to_string());
    }

    if (unlikely(input.shape()[0] != indices.shape()[0]))
    {
        THROW_INVALID_ARG("gather: batch size mismatch. input has {} samples, indices has {} samples",
                          input.shape()[0], indices.shape()[0]);
    }

    VALIDATE_SAME_CUDA_DEVICE(input, indices);

    size_t N = input.shape()[0];
    size_t C = input.shape()[1];

    // 验证索引范围：将索引复制到 CPU 进行验证
    // 这对于错误检查是必要的，因为 CUDA kernel 无法抛出异常
    std::vector<int32_t> indices_cpu_int32;
    std::vector<int64_t> indices_cpu_int64;
    if (indices.dtype() == DataType::kInt32)
    {
        indices_cpu_int32.resize(N);
        CUDA_CHECK(cudaMemcpy(indices_cpu_int32.data(), indices.storage()->data(), N * sizeof(int32_t), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < N; ++i)
        {
            if (unlikely(indices_cpu_int32[i] < 0 || indices_cpu_int32[i] >= static_cast<int32_t>(C)))
            {
                THROW_INVALID_ARG("gather: index {} out of range [0, {})", indices_cpu_int32[i], C);
            }
        }
    }
    else if (indices.dtype() == DataType::kInt64)
    {
        indices_cpu_int64.resize(N);
        CUDA_CHECK(cudaMemcpy(indices_cpu_int64.data(), indices.storage()->data(), N * sizeof(int64_t), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < N; ++i)
        {
            if (unlikely(indices_cpu_int64[i] < 0 || indices_cpu_int64[i] >= static_cast<int64_t>(C)))
            {
                THROW_INVALID_ARG("gather: index {} out of range [0, {})", indices_cpu_int64[i], C);
            }
        }
    }

    // 创建输出矩阵
    Shape output_shape{N};
    auto result = std::make_unique<OriginMat>(output_shape, input.dtype(), input.device());

    const void *input_data  = input.storage()->data();
    const void *indices_data = indices.storage()->data();
    void *output_data       = result->storage()->data();

    // 计算线程块和网格大小
    const size_t threads_per_block = 256;
    const size_t num_blocks        = (N + threads_per_block - 1) / threads_per_block;

    // 使用类型分发器执行 gather 操作
    device_common::TypeDispatcher::dispatch_void(input.dtype(), [&]<typename T>() {
        if (indices.dtype() == DataType::kInt32)
        {
            gather_kernel_int32<T><<<num_blocks, threads_per_block>>>(
                static_cast<const T *>(input_data), static_cast<const int32_t *>(indices_data),
                static_cast<T *>(output_data), N, C);
        }
        else if (indices.dtype() == DataType::kInt64)
        {
            gather_kernel_int64<T><<<num_blocks, threads_per_block>>>(
                static_cast<const T *>(input_data), static_cast<const int64_t *>(indices_data),
                static_cast<T *>(output_data), N, C);
        }
        else
        {
            THROW_INVALID_ARG("gather: indices must be int32 or int64, but got {}", dtype_to_string(indices.dtype()));
        }
    });

    CUDA_CHECK_ASYNC();
    return result;
}

/**
 * @brief CUDA one_hot：将索引转换为 one-hot 编码
 * @param indices 索引向量 (N,)，每个元素在 [0, num_classes) 范围内
 * @param num_classes 类别数量
 * @return one-hot 编码矩阵 (N, num_classes)
 */
std::unique_ptr<Mat> one_hot(const OriginMat &indices, int num_classes)
{
    if (unlikely(indices.shape().size() != 1))
    {
        THROW_INVALID_ARG("one_hot: indices must be 1D (N,), but got shape {}", indices.shape().to_string());
    }

    if (unlikely(num_classes <= 0))
    {
        THROW_INVALID_ARG("one_hot: num_classes must be positive, but got {}", num_classes);
    }

    size_t N = indices.shape()[0];

    // 创建输出矩阵
    Shape output_shape{N, static_cast<size_t>(num_classes)};
    auto result = std::make_unique<OriginMat>(output_shape, DataType::kFloat32, indices.device());

    const void *indices_data = indices.storage()->data();
    void *output_data        = result->storage()->data();

    // 初始化为0
    size_t data_size = N * num_classes * sizeof(float);
    CUDA_CHECK(cudaMemset(output_data, 0, data_size));

    // 计算线程块和网格大小
    const size_t threads_per_block = 256;
    const size_t num_blocks        = (N + threads_per_block - 1) / threads_per_block;

    // 根据索引类型启动对应的 kernel
    if (indices.dtype() == DataType::kInt32)
    {
        one_hot_kernel_int32<<<num_blocks, threads_per_block>>>(
            static_cast<const int32_t *>(indices_data), static_cast<float *>(output_data), N, num_classes);
    }
    else if (indices.dtype() == DataType::kInt64)
    {
        one_hot_kernel_int64<<<num_blocks, threads_per_block>>>(
            static_cast<const int64_t *>(indices_data), static_cast<float *>(output_data), N, num_classes);
    }
    else
    {
        THROW_INVALID_ARG("one_hot: indices must be int32 or int64, but got {}", dtype_to_string(indices.dtype()));
    }

    CUDA_CHECK_ASYNC();
    return result;
}

}  // namespace cuda
}  // namespace origin
