#include <cuda_runtime.h>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

// Div Operator Performance Comparison
// ===================================================================================
// Shape           Repeat   Device   Dtype     OriginDL(us)    PyTorch(us)     Speedup
// -----------------------------------------------------------------------------------
// {1,1}           100      cuda:0   float32   7.4600          11.9916         1.6075 
// {10,10}         100      cuda:0   float32   7.3600          11.6907         1.5884 
// {100,100}       100      cuda:0   float32   7.3100          11.5103         1.5746 
// {1000,1000}     100      cuda:0   float32   7.2600          11.9496         1.6460 
// {10000,10000}   100      cuda:0   float32   884.2900        887.0096        1.0031 
// ===================================================================================

namespace origin
{
namespace cuda
{

/**
 * @brief 元素级除法kernel（相同形状）- 最朴素实现
 */
template <typename T>
__global__ void divide_elementwise_native_kernel(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
                                                size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] / B[i];
    }
}

/**
 * @brief 向量化元素级除法kernel - float4版本
 */
__global__ void divide_elementwise_vectorized_float4_kernel(const float *__restrict__ A,
                                                           const float *__restrict__ B,
                                                           float *__restrict__ C,
                                                           size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        float4 vec_a = *reinterpret_cast<const float4 *>(&A[vector_idx]);
        float4 vec_b = *reinterpret_cast<const float4 *>(&B[vector_idx]);
        float4 vec_c = make_float4(vec_a.x / vec_b.x, vec_a.y / vec_b.y, vec_a.z / vec_b.z, vec_a.w / vec_b.w);
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        size_t base_idx = vector_idx;
#pragma unroll(VECTOR_SIZE)
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            C[base_idx + i] = A[base_idx + i] / B[base_idx + i];
        }
    }
}

/**
 * @brief 广播除法kernel - B是标量
 */
template <typename T>
__global__ void divide_broadcast_kernel(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
                                        size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] / B[0];
    }
}

/**
 * @brief 向量化广播除法kernel - float4版本，B是标量
 */
__global__ void divide_broadcast_vectorized_float4_kernel(const float *__restrict__ A,
                                                         const float *__restrict__ B,
                                                         float *__restrict__ C,
                                                         size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        float4 vec_a   = *reinterpret_cast<const float4 *>(&A[vector_idx]);
        float scalar_b = B[0];
        float4 vec_b   = make_float4(scalar_b, scalar_b, scalar_b, scalar_b);
        float4 vec_c   = make_float4(vec_a.x / vec_b.x, vec_a.y / vec_b.y, vec_a.z / vec_b.z, vec_a.w / vec_b.w);
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        size_t base_idx = vector_idx;
#pragma unroll(VECTOR_SIZE)
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            C[base_idx + i] = A[base_idx + i] / B[0];
        }
    }
}

/** @brief 标量除向量kernel - A是标量，C[i]=A[0]/B[i] */
template <typename T>
__global__ void divide_scalar_div_vector_kernel(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
                                                size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[0] / B[i];
}

/** @brief 向量化标量除向量kernel - float4版本，A是标量，C[i]=A[0]/B[i] */
__global__ void divide_scalar_div_vector_vectorized_float4_kernel(const float *__restrict__ A,
                                                                  const float *__restrict__ B,
                                                                  float *__restrict__ C,
                                                                  size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N         = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx           = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;
    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        float scalar_a = A[0];
        float4 vec_b   = *reinterpret_cast<const float4 *>(&B[vector_idx]);
        float4 vec_c   = make_float4(scalar_a / vec_b.x, scalar_a / vec_b.y, scalar_a / vec_b.z, scalar_a / vec_b.w);
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        size_t base_idx = vector_idx;
#pragma unroll(VECTOR_SIZE)
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
            C[base_idx + i] = A[0] / B[base_idx + i];
    }
}

/**
 * @brief CUDA除法算子统一实现
 */
std::unique_ptr<Mat> divide(const OriginMat &a, const OriginMat &b, OriginMat *out)
{
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CUDA_DEVICE(a, b);

    Shape result_shape = origin::utils::compute::compute_broadcast_shape(a, b);

    OriginMat *result_ptr = nullptr;
    std::unique_ptr<OriginMat> result_unique;

    if (out != nullptr)
    {
        if (unlikely(out->shape() != result_shape || out->dtype() != a.dtype() || out->device() != a.device()))
        {
            THROW_INVALID_ARG(
                "Output tensor mismatch. Expected shape={}, dtype={}, device={}, but got shape={}, "
                "dtype={}, device={}",
                result_shape.to_string(), dtype_to_string(a.dtype()), a.device().to_string(), out->shape().to_string(),
                dtype_to_string(out->dtype()), out->device().to_string());
        }
        result_ptr = out;
    }
    else
    {
        result_unique = std::make_unique<OriginMat>(result_shape, a.dtype(), a.device());
        result_ptr    = result_unique.get();
    }

    const void *a_data = a.storage()->data();
    const void *b_data = b.storage()->data();
    void *c_data       = result_ptr->storage()->data();

    if (a.shape() == b.shape())
    {
        const size_t num_elements = a.elements();
        if (a.dtype() == DataType::kFloat32)
        {
            constexpr size_t VECTOR_SIZE       = 4;
            const size_t threads_per_block    = 256;
            const size_t vectorized_elements  = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
            const size_t num_blocks           = (vectorized_elements + threads_per_block - 1) / threads_per_block;
            divide_elementwise_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
                static_cast<const float *>(a_data), static_cast<const float *>(b_data), static_cast<float *>(c_data),
                num_elements);
        }
        else
        {
            const size_t threads_per_block = 256;
            const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
            device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
                divide_elementwise_native_kernel<T><<<num_blocks, threads_per_block>>>(
                    static_cast<const T *>(a_data), static_cast<const T *>(b_data), static_cast<T *>(c_data),
                    num_elements);
            });
        }
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        const size_t num_elements = result_ptr->elements();
        if (a.elements() == 1)
        {
            // 标量 / 向量：C[i] = A[0] / B[i]
            if (a.dtype() == DataType::kFloat32)
            {
                constexpr size_t VECTOR_SIZE      = 4;
                const size_t threads_per_block   = 256;
                const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
                const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
                divide_scalar_div_vector_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
                    static_cast<const float *>(a_data), static_cast<const float *>(b_data),
                    static_cast<float *>(c_data), num_elements);
            }
            else
            {
                const size_t threads_per_block = 256;
                const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
                device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
                    divide_scalar_div_vector_kernel<T><<<num_blocks, threads_per_block>>>(
                        static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                        static_cast<T *>(c_data), num_elements);
                });
            }
        }
        else
        {
            // 向量 / 标量：C[i] = A[i] / B[0]
            const void *vec_data    = a_data;
            const void *scalar_data = b_data;
            if (a.dtype() == DataType::kFloat32)
            {
                constexpr size_t VECTOR_SIZE      = 4;
                const size_t threads_per_block   = 256;
                const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
                const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
                divide_broadcast_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
                    static_cast<const float *>(vec_data), static_cast<const float *>(scalar_data),
                    static_cast<float *>(c_data), num_elements);
            }
            else
            {
                const size_t threads_per_block = 256;
                const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
                device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
                    divide_broadcast_kernel<T><<<num_blocks, threads_per_block>>>(
                        static_cast<const T *>(vec_data), static_cast<const T *>(scalar_data),
                        static_cast<T *>(c_data), num_elements);
                });
            }
        }
    }
    else
    {
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CUDA divide operation");
    }

    return result_unique;
}

}  // namespace cuda
}  // namespace origin
