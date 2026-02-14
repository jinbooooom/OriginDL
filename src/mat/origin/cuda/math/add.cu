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

// $ python3 run_benchmark.py -d cuda -f add

// ===================================================================================
// Add Operator Performance Comparison
// ===================================================================================
// Shape           Repeat   Device   Dtype     OriginDL(us)    PyTorch(us)     Speedup
// -----------------------------------------------------------------------------------
// {1,1}           100      cuda:0   float32   6.5100          12.5775         1.9320 
// {10,10}         100      cuda:0   float32   6.2600          12.2546         1.9576 
// {100,100}       100      cuda:0   float32   6.2900          12.7134         2.0212 
// {1000,1000}     100      cuda:0   float32   7.5300          12.5128         1.6617 
// {10000,10000}   100      cuda:0   float32   857.7500        858.8290        1.0013 
// ===================================================================================
namespace origin
{
namespace cuda
{

/**
 * @brief 元素级加法kernel（相同形状）- 最朴素实现
 * @details 每个线程处理一个元素的加法运算，用于不支持向量化的类型或边界情况
 */
template <typename T>
__global__ void add_elementwise_native_kernel(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief 向量化元素级加法kernel - float4版本
 * @details 每个线程使用float4一次处理4个float元素，提高内存带宽利用率
 */
__global__ void add_elementwise_vectorized_float4_kernel(const float *__restrict__ A,
                                                         const float *__restrict__ B,
                                                         float *__restrict__ C,
                                                         size_t N)
{
    // 计算向量化的元素数量（每个float4包含4个float）
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    // 向量化处理主体部分
    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        // 使用float4一次加载4个float
        float4 vec_a = *reinterpret_cast<const float4 *>(&A[vector_idx]);
        float4 vec_b = *reinterpret_cast<const float4 *>(&B[vector_idx]);

        // 执行向量化加法
        float4 vec_c = make_float4(vec_a.x + vec_b.x, vec_a.y + vec_b.y, vec_a.z + vec_b.z, vec_a.w + vec_b.w);

        // 使用float4一次存储4个float
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        // 处理边界情况：逐个处理剩余元素
        size_t base_idx = vector_idx;
#pragma unroll(VECTOR_SIZE)
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            C[base_idx + i] = A[base_idx + i] + B[base_idx + i];
        }
    }
}

/**
 * @brief 广播加法kernel - B是标量
 * @details A和C是长度为N的向量，B是标量（长度为1），计算 C[i] = A[i] + B[0]
 */
template <typename T>
__global__ void add_broadcast_kernel(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[0];
    }
}

/**
 * @brief 向量化广播加法kernel - float4版本，B是标量
 * @details A和C是长度为N的向量，B是标量（长度为1），使用float4向量化优化
 *          计算 C[i:i+4] = A[i:i+4] + B[0]
 */
__global__ void add_broadcast_vectorized_float4_kernel(const float *__restrict__ A,
                                                       const float *__restrict__ B,
                                                       float *__restrict__ C,
                                                       size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    // 向量化处理主体部分
    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        // A是向量：向量化加载
        float4 vec_a = *reinterpret_cast<const float4 *>(&A[vector_idx]);
        // B是标量：广播B[0]到float4
        float scalar_b = B[0];
        float4 vec_b   = make_float4(scalar_b, scalar_b, scalar_b, scalar_b);

        // 执行向量化加法
        float4 vec_c = make_float4(vec_a.x + vec_b.x, vec_a.y + vec_b.y, vec_a.z + vec_b.z, vec_a.w + vec_b.w);

        // 向量化存储
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        // 处理边界情况：逐个处理剩余元素
        size_t base_idx = vector_idx;
#pragma unroll(VECTOR_SIZE)
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            C[base_idx + i] = A[base_idx + i] + B[0];
        }
    }
}

/**
 * @brief CUDA加法算子统一实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵返回新矩阵，否则将结果写入out，返回nullptr
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b, OriginMat *out)
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
        // 相同形状：直接元素级运算（最常见）
        const size_t num_elements = a.elements();
        if (a.dtype() == DataType::kFloat32)  // float32 类型是最常见的
        {
            // float4向量化版本：每个线程处理4个元素
            constexpr size_t VECTOR_SIZE     = 4;
            const size_t threads_per_block   = 256;
            const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
            const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
            add_elementwise_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
                static_cast<const float *>(a_data), static_cast<const float *>(b_data), static_cast<float *>(c_data),
                num_elements);
        }
        else
        {
            const size_t threads_per_block = 256;
            const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
            device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
                add_elementwise_native_kernel<T><<<num_blocks, threads_per_block>>>(static_cast<const T *>(a_data),
                                                                             static_cast<const T *>(b_data),
                                                                             static_cast<T *>(c_data), num_elements);
            });
        }
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        // 简单广播：一个操作数是标量（次常见）。统一成向量在左、标量在右再 dispatch。
        const size_t num_elements = result_ptr->elements();
        const void *vec_data    = (a.elements() == 1) ? b_data : a_data;
        const void *scalar_data = (a.elements() == 1) ? a_data : b_data;

        if (a.dtype() == DataType::kFloat32)
        {
            constexpr size_t VECTOR_SIZE       = 4;
            const size_t threads_per_block    = 256;
            const size_t vectorized_elements  = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
            const size_t num_blocks           = (vectorized_elements + threads_per_block - 1) / threads_per_block;
            add_broadcast_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
                static_cast<const float *>(vec_data), static_cast<const float *>(scalar_data),
                static_cast<float *>(c_data), num_elements);
        }
        else
        {
            const size_t threads_per_block = 256;
            const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
            device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
                add_broadcast_kernel<T><<<num_blocks, threads_per_block>>>(
                    static_cast<const T *>(vec_data), static_cast<const T *>(scalar_data),
                    static_cast<T *>(c_data), num_elements);
            });
        }
    }
    else
    {
        // 复杂广播：需要计算步长信息
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CUDA add operation");
    }

    return result_unique;
}

}  // namespace cuda
}  // namespace origin
