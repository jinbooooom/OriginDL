#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/operation_templates.h"
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
 * @brief Sigmoid kernel（基础版本）：y = 1 / (1 + exp(-x))
 */
template <typename T>
__global__ void sigmoid_kernel(const T *__restrict__ A, T *__restrict__ C, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            C[i] = 1.0f / (1.0f + expf(-A[i]));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            C[i] = 1.0 / (1.0 + exp(-A[i]));
        }
        else
        {
            C[i] = T(1) / (T(1) + std::exp(-A[i]));
        }
    }
}

/**
 * @brief 向量化 Sigmoid kernel - float4 版本
 */
__global__ void sigmoid_vectorized_float4_kernel(const float *__restrict__ A, float *__restrict__ C, size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        float4 vec_a = *reinterpret_cast<const float4 *>(&A[vector_idx]);
        float4 vec_c;
        vec_c.x                                     = 1.0f / (1.0f + expf(-vec_a.x));
        vec_c.y                                     = 1.0f / (1.0f + expf(-vec_a.y));
        vec_c.z                                     = 1.0f / (1.0f + expf(-vec_a.z));
        vec_c.w                                     = 1.0f / (1.0f + expf(-vec_a.w));
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        size_t base_idx = vector_idx;
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            C[base_idx + i] = 1.0f / (1.0f + expf(-A[base_idx + i]));
        }
    }
}

/**
 * @brief Sigmoid 反向传播 kernel：gx = gy * y * (1 - y)
 */
template <typename T>
__global__ void sigmoid_backward_kernel(const T *__restrict__ gy, const T *__restrict__ y, T *__restrict__ gx, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        gx[i] = gy[i] * y[i] * (T(1) - y[i]);
    }
}

__global__ void sigmoid_backward_vectorized_float4_kernel(const float *__restrict__ gy,
                                                          const float *__restrict__ y,
                                                          float *__restrict__ gx,
                                                          size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        float4 v_gy = *reinterpret_cast<const float4 *>(&gy[vector_idx]);
        float4 v_y  = *reinterpret_cast<const float4 *>(&y[vector_idx]);
        float4 v_gx;
        v_gx.x                                       = v_gy.x * v_y.x * (1.0f - v_y.x);
        v_gx.y                                       = v_gy.y * v_y.y * (1.0f - v_y.y);
        v_gx.z                                       = v_gy.z * v_y.z * (1.0f - v_y.z);
        v_gx.w                                       = v_gy.w * v_y.w * (1.0f - v_y.w);
        *reinterpret_cast<float4 *>(&gx[vector_idx]) = v_gx;
    }
    else
    {
        size_t base_idx = vector_idx;
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            size_t j = base_idx + i;
            gx[j]    = gy[j] * y[j] * (1.0f - y[j]);
        }
    }
}

std::unique_ptr<Mat> sigmoid(const OriginMat &mat, OriginMat *out)
{
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute Sigmoid of empty matrix");
    }
    VALIDATE_CUDA_DEVICE(mat);

    OriginMat *result_ptr = nullptr;
    std::unique_ptr<OriginMat> result_unique;

    if (out != nullptr)
    {
        if (unlikely(out->shape() != mat.shape() || out->dtype() != mat.dtype() || out->device() != mat.device()))
        {
            THROW_INVALID_ARG(
                "Output tensor mismatch. Expected shape={}, dtype={}, device={}, but got shape={}, "
                "dtype={}, device={}",
                mat.shape().to_string(), dtype_to_string(mat.dtype()), mat.device().to_string(),
                out->shape().to_string(), dtype_to_string(out->dtype()), out->device().to_string());
        }
        result_ptr = out;
    }
    else
    {
        result_unique = std::make_unique<OriginMat>(mat.shape(), mat.dtype(), mat.device());
        result_ptr    = result_unique.get();
    }

    const void *a_data        = mat.storage()->data();
    void *c_data              = result_ptr->storage()->data();
    const size_t num_elements = mat.elements();

    if (mat.dtype() == DataType::kFloat32)
    {
        constexpr size_t VECTOR_SIZE     = 4;
        const size_t threads_per_block   = 256;
        const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
        const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
        sigmoid_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(static_cast<const float *>(a_data),
                                                                            static_cast<float *>(c_data), num_elements);
    }
    else
    {
        const size_t threads_per_block = 256;
        const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            sigmoid_kernel<T><<<num_blocks, threads_per_block>>>(static_cast<const T *>(a_data),
                                                                 static_cast<T *>(c_data), num_elements);
        });
    }

    return result_unique;
}

std::unique_ptr<Mat> sigmoid_backward(const OriginMat &gy, const OriginMat &y)
{
    if (unlikely(gy.elements() == 0) || unlikely(y.elements() != gy.elements()))
    {
        THROW_INVALID_ARG("sigmoid_backward: gy and y must have same non-zero size");
    }
    VALIDATE_CUDA_DEVICE(gy);
    VALIDATE_CUDA_DEVICE(y);
    if (unlikely(gy.shape() != y.shape() || gy.dtype() != y.dtype()))
    {
        THROW_INVALID_ARG("sigmoid_backward: gy and y must have same shape and dtype");
    }

    auto result         = std::make_unique<OriginMat>(gy.shape(), gy.dtype(), gy.device());
    const void *gy_data = gy.storage()->data();
    const void *y_data  = y.storage()->data();
    void *gx_data       = result->storage()->data();
    const size_t n      = gy.elements();

    if (gy.dtype() == DataType::kFloat32)
    {
        constexpr size_t VECTOR_SIZE     = 4;
        const size_t threads_per_block   = 256;
        const size_t vectorized_elements = (n + VECTOR_SIZE - 1) / VECTOR_SIZE;
        const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
        sigmoid_backward_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
            static_cast<const float *>(gy_data), static_cast<const float *>(y_data), static_cast<float *>(gx_data), n);
    }
    else
    {
        const size_t threads_per_block = 256;
        const size_t num_blocks        = (n + threads_per_block - 1) / threads_per_block;
        device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
            sigmoid_backward_kernel<T><<<num_blocks, threads_per_block>>>(
                static_cast<const T *>(gy_data), static_cast<const T *>(y_data), static_cast<T *>(gx_data), n);
        });
    }

    return result;
}

}  // namespace cuda
}  // namespace origin
