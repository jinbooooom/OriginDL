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
 * @brief ReLU kernel（基础版本）
 * @details 每个线程处理一个元素的 ReLU 运算，用于不支持向量化的类型或边界情况
 */
template <typename T>
__global__ void relu_kernel(const T *__restrict__ A, T *__restrict__ C, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        // 使用 fmaxf 避免分支（对于 float 类型）
        // 对于其他类型，使用三元运算符
        if constexpr (std::is_same_v<T, float>)
        {
            C[i] = fmaxf(0.0f, A[i]);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            C[i] = fmax(0.0, A[i]);
        }
        else
        {
            C[i] = (A[i] > T(0)) ? A[i] : T(0);
        }
    }
}

/**
 * @brief 向量化 ReLU kernel - float4版本
 * @details 每个线程使用float4一次处理4个float元素，使用fmaxf避免分支，提高内存带宽利用率和性能
 */
__global__ void relu_vectorized_float4_kernel(const float *__restrict__ A, float *__restrict__ C, size_t N)
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

        // 使用fmaxf避免分支（fmaxf(0, x) = max(0, x) = ReLU(x)）
        float4 vec_c;
        vec_c.x = fmaxf(0.0f, vec_a.x);
        vec_c.y = fmaxf(0.0f, vec_a.y);
        vec_c.z = fmaxf(0.0f, vec_a.z);
        vec_c.w = fmaxf(0.0f, vec_a.w);

        // 使用float4一次存储4个float
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        // 处理边界情况：逐个处理剩余元素
        size_t base_idx = vector_idx;
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            C[base_idx + i] = fmaxf(0.0f, A[base_idx + i]);
        }
    }
}

/**
 * @brief CUDA ReLU 激活函数统一实现
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> relu(const OriginMat &mat, OriginMat *out)
{
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute ReLU of empty matrix");
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

    const void *a_data = mat.storage()->data();
    void *c_data       = result_ptr->storage()->data();

    const size_t num_elements = mat.elements();

    // float32 类型是最常见的，使用 float4 向量化优化
    if (mat.dtype() == DataType::kFloat32)
    {
        // float4向量化版本：每个线程处理4个元素
        constexpr size_t VECTOR_SIZE     = 4;
        const size_t threads_per_block   = 256;
        const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
        const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
        relu_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
            static_cast<const float *>(a_data), static_cast<float *>(c_data), num_elements);
    }
    else
    {
        // 其他类型使用基础版本
        const size_t threads_per_block = 256;
        const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            relu_kernel<T><<<num_blocks, threads_per_block>>>(static_cast<const T *>(a_data),
                                                               static_cast<T *>(c_data), num_elements);
        });
    }

    return result_unique;
}

}  // namespace cuda
}  // namespace origin
