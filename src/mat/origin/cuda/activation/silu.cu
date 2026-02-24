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
 * @brief SiLU kernel：y = x * sigmoid(x) = x / (1 + exp(-x))
 */
template <typename T>
__global__ void silu_native_kernel(const T *__restrict__ x, T *__restrict__ y, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        const T v = x[i];
        if constexpr (std::is_same_v<T, float>)
        {
            y[i] = v / (T(1) + expf(-v));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            y[i] = v / (T(1) + exp(-v));
        }
        else
        {
            const T s = T(1) / (T(1) + std::exp(-v));
            y[i]      = v * s;
        }
    }
}

__global__ void silu_vectorized_float4_kernel(const float *__restrict__ x, float *__restrict__ y, size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        float4 vec_x = *reinterpret_cast<const float4 *>(&x[vector_idx]);
        float4 vec_y;
        vec_y.x = vec_x.x / (1.0f + expf(-vec_x.x));
        vec_y.y = vec_x.y / (1.0f + expf(-vec_x.y));
        vec_y.z = vec_x.z / (1.0f + expf(-vec_x.z));
        vec_y.w = vec_x.w / (1.0f + expf(-vec_x.w));
        *reinterpret_cast<float4 *>(&y[vector_idx]) = vec_y;
    }
    else
    {
        size_t base_idx = vector_idx;
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            const size_t j = base_idx + i;
            const float v  = x[j];
            y[j]           = v / (1.0f + expf(-v));
        }
    }
}

/**
 * @brief SiLU 反向 kernel：gx = gy * (s + x * s * (1 - s))，其中 s = sigmoid(x)
 */
template <typename T>
__global__ void silu_backward_native_kernel(const T *__restrict__ gy,
                                     const T *__restrict__ x,
                                     T *__restrict__ gx,
                                     size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        const T v = x[i];
        T s;
        if constexpr (std::is_same_v<T, float>)
        {
            s = T(1) / (T(1) + expf(-v));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            s = T(1) / (T(1) + exp(-v));
        }
        else
        {
            s = T(1) / (T(1) + std::exp(-v));
        }
        const T grad_silu = s + v * s * (T(1) - s);
        gx[i]             = gy[i] * grad_silu;
    }
}

std::unique_ptr<Mat> silu(const OriginMat &mat, OriginMat *out)
{
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute SiLU of empty matrix");
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

    const void *x_data       = mat.storage()->data();
    void *y_data             = result_ptr->storage()->data();
    const size_t num_elements = mat.elements();

    if (mat.dtype() == DataType::kFloat32)
    {
        constexpr size_t VECTOR_SIZE     = 4;
        const size_t threads_per_block   = 256;
        const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
        const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
        silu_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
            static_cast<const float *>(x_data), static_cast<float *>(y_data), num_elements);
    }
    else
    {
        const size_t threads_per_block = 256;
        const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            silu_native_kernel<T><<<num_blocks, threads_per_block>>>(
                static_cast<const T *>(x_data), static_cast<T *>(y_data), num_elements);
        });
    }

    return result_unique;
}

std::unique_ptr<Mat> silu_backward(const OriginMat &gy, const OriginMat &x)
{
    if (unlikely(gy.elements() == 0) || unlikely(x.elements() != gy.elements()))
    {
        THROW_INVALID_ARG("silu_backward: gy and x must have same non-zero size");
    }
    VALIDATE_CUDA_DEVICE(gy);
    VALIDATE_CUDA_DEVICE(x);
    if (unlikely(gy.shape() != x.shape() || gy.dtype() != x.dtype()))
    {
        THROW_INVALID_ARG("silu_backward: gy and x must have same shape and dtype");
    }

    auto result   = std::make_unique<OriginMat>(gy.shape(), gy.dtype(), gy.device());
    const void *gy_data = gy.storage()->data();
    const void *x_data  = x.storage()->data();
    void *gx_data      = result->storage()->data();
    const size_t n     = gy.elements();

    const size_t threads_per_block = 256;
    const size_t num_blocks        = (n + threads_per_block - 1) / threads_per_block;
    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        silu_backward_native_kernel<T><<<num_blocks, threads_per_block>>>(
            static_cast<const T *>(gy_data), static_cast<const T *>(x_data), static_cast<T *>(gx_data), n);
    });

    return result;
}

}  // namespace cuda
}  // namespace origin

