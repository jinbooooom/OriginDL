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

namespace origin
{
namespace cuda
{

// ==================== CUDA Kernels ====================

/**
 * @brief 元素级比较kernel - 相同形状
 * @tparam T 数据类型
 * @tparam Op 比较操作类型
 * @details A、B和C是长度为N的向量，计算 C[i] = Op(A[i], B[i])
 */
template <typename T, typename Op>
__global__ void compare_elementwise_kernel(const T *__restrict__ A,
                                           const T *__restrict__ B,
                                           T *__restrict__ C,
                                           size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        Op op;
        C[i] = op(A[i], B[i]);
    }
}

/**
 * @brief 向量化元素级比较kernel - float4版本
 * @tparam Op 比较操作类型
 * @details 每个线程使用float4一次处理4个float元素
 */
template <typename Op>
__global__ void compare_elementwise_vectorized_float4_kernel(const float *__restrict__ A,
                                                             const float *__restrict__ B,
                                                             float *__restrict__ C,
                                                             size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    Op op;

    // 向量化处理主体部分
    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        // 使用float4一次加载4个float
        float4 vec_a = *reinterpret_cast<const float4 *>(&A[vector_idx]);
        float4 vec_b = *reinterpret_cast<const float4 *>(&B[vector_idx]);

        // 执行向量化比较
        float4 vec_c;
        vec_c.x = op(vec_a.x, vec_b.x);
        vec_c.y = op(vec_a.y, vec_b.y);
        vec_c.z = op(vec_a.z, vec_b.z);
        vec_c.w = op(vec_a.w, vec_b.w);

        // 使用float4一次存储4个float
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        // 处理边界情况：逐个处理剩余元素
        size_t base_idx = vector_idx;
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            C[base_idx + i] = op(A[base_idx + i], B[base_idx + i]);
        }
    }
}

/**
 * @brief 广播比较kernel - threshold是标量
 * @tparam T 数据类型
 * @tparam Op 比较操作类型
 * @details A和C是长度为N的向量，threshold是标量（长度为1），计算 C[i] = Op(A[i], threshold[0])
 */
template <typename T, typename Op>
__global__ void compare_broadcast_kernel(const T *__restrict__ A,
                                         const T *__restrict__ threshold,
                                         T *__restrict__ C,
                                         size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        Op op;
        C[i] = op(A[i], threshold[0]);
    }
}

/**
 * @brief 向量化广播比较kernel - float4版本，threshold是标量
 * @tparam Op 比较操作类型
 * @details A和C是长度为N的向量，threshold是标量（长度为1），使用float4向量化优化
 *          计算 C[i:i+4] = Op(A[i:i+4], threshold[0])
 */
template <typename Op>
__global__ void compare_broadcast_vectorized_float4_kernel(const float *__restrict__ A,
                                                           const float *__restrict__ threshold,
                                                           float *__restrict__ C,
                                                           size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    Op op;
    float scalar_threshold = threshold[0];

    // 向量化处理主体部分
    if (vector_idx + VECTOR_SIZE <= vectorized_N)
    {
        // A是向量：向量化加载
        float4 vec_a = *reinterpret_cast<const float4 *>(&A[vector_idx]);

        // 执行向量化比较
        float4 vec_c;
        vec_c.x = op(vec_a.x, scalar_threshold);
        vec_c.y = op(vec_a.y, scalar_threshold);
        vec_c.z = op(vec_a.z, scalar_threshold);
        vec_c.w = op(vec_a.w, scalar_threshold);

        // 向量化存储
        *reinterpret_cast<float4 *>(&C[vector_idx]) = vec_c;
    }
    else
    {
        // 处理边界情况：逐个处理剩余元素
        size_t base_idx = vector_idx;
        for (size_t i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i)
        {
            C[base_idx + i] = op(A[base_idx + i], scalar_threshold);
        }
    }
}

// ==================== 实现函数 ====================

/**
 * @brief CUDA比较算子统一实现
 * @tparam Op 比较操作类型
 * @param mat 输入矩阵
 * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
template <typename Op>
std::unique_ptr<Mat> compare_impl(const OriginMat &mat, const OriginMat &threshold, OriginMat *out)
{
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute comparison of empty matrix");
    }
    VALIDATE_SAME_DTYPE(mat, threshold);
    VALIDATE_SAME_CUDA_DEVICE(mat, threshold);

    // 检查 threshold 是否为标量（元素数量为 1）
    bool is_scalar_threshold = threshold.elements() == 1;

    Shape result_shape;
    if (mat.shape() == threshold.shape())
    {
        // 相同形状：结果形状与输入相同
        result_shape = mat.shape();
    }
    else if (is_scalar_threshold)
    {
        // 标量广播：结果形状与 mat 相同
        result_shape = mat.shape();
    }
    else
    {
        THROW_INVALID_ARG(
            "Compare operator: threshold must be scalar (shape {{}} or {{1}}) or have same shape as input. "
            "Got mat shape={}, threshold shape={}",
            mat.shape().to_string(), threshold.shape().to_string());
    }

    OriginMat *result_ptr = nullptr;
    std::unique_ptr<OriginMat> result_unique;

    if (out != nullptr)
    {
        if (unlikely(out->shape() != result_shape || out->dtype() != mat.dtype() || out->device() != mat.device()))
        {
            THROW_INVALID_ARG(
                "Output tensor mismatch. Expected shape={}, dtype={}, device={}, but got shape={}, "
                "dtype={}, device={}",
                result_shape.to_string(), dtype_to_string(mat.dtype()), mat.device().to_string(),
                out->shape().to_string(), dtype_to_string(out->dtype()), out->device().to_string());
        }
        result_ptr = out;
    }
    else
    {
        result_unique = std::make_unique<OriginMat>(result_shape, mat.dtype(), mat.device());
        result_ptr    = result_unique.get();
    }

    const void *mat_data       = mat.storage()->data();
    const void *threshold_data = threshold.storage()->data();
    void *result_data          = result_ptr->storage()->data();

    // 分支优化 - 参考 add.cu 的实现方式
    if (mat.shape() == threshold.shape())
    {
        // 相同形状：直接元素级运算（最常见）
        const size_t num_elements = mat.elements();
        if (mat.dtype() == DataType::kFloat32)  // float32 类型是最常见的
        {
            // float4向量化版本：每个线程处理4个元素
            constexpr size_t VECTOR_SIZE     = 4;
            const size_t threads_per_block   = 256;
            const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
            const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
            compare_elementwise_vectorized_float4_kernel<Op><<<num_blocks, threads_per_block>>>(
                static_cast<const float *>(mat_data), static_cast<const float *>(threshold_data),
                static_cast<float *>(result_data), num_elements);
        }
        else
        {
            const size_t threads_per_block = 256;
            const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
            device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
                compare_elementwise_kernel<T, Op><<<num_blocks, threads_per_block>>>(
                    static_cast<const T *>(mat_data), static_cast<const T *>(threshold_data),
                    static_cast<T *>(result_data), num_elements);
            });
        }
    }
    else if (is_scalar_threshold)
    {
        // 简单广播：threshold是标量（次常见）
        const size_t num_elements = result_ptr->elements();

        // 仅对float类型使用向量化优化，其他类型使用基础版本
        if (mat.dtype() == DataType::kFloat32)
        {
            constexpr size_t VECTOR_SIZE     = 4;
            const size_t threads_per_block   = 256;
            const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
            const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
            compare_broadcast_vectorized_float4_kernel<Op><<<num_blocks, threads_per_block>>>(
                static_cast<const float *>(mat_data), static_cast<const float *>(threshold_data),
                static_cast<float *>(result_data), num_elements);
        }
        else
        {
            const size_t threads_per_block = 256;
            const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
            device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
                compare_broadcast_kernel<T, Op><<<num_blocks, threads_per_block>>>(
                    static_cast<const T *>(mat_data), static_cast<const T *>(threshold_data),
                    static_cast<T *>(result_data), num_elements);
            });
        }
    }

    return result_unique;
}

// 显式实例化所有比较操作
std::unique_ptr<Mat> eq(const OriginMat &mat, const OriginMat &threshold, OriginMat *out)
{
    return compare_impl<EqualOp>(mat, threshold, out);
}

std::unique_ptr<Mat> ne(const OriginMat &mat, const OriginMat &threshold, OriginMat *out)
{
    return compare_impl<NotEqualOp>(mat, threshold, out);
}

std::unique_ptr<Mat> lt(const OriginMat &mat, const OriginMat &threshold, OriginMat *out)
{
    return compare_impl<LessThanOp>(mat, threshold, out);
}

std::unique_ptr<Mat> le(const OriginMat &mat, const OriginMat &threshold, OriginMat *out)
{
    return compare_impl<LessEqualOp>(mat, threshold, out);
}

std::unique_ptr<Mat> gt(const OriginMat &mat, const OriginMat &threshold, OriginMat *out)
{
    return compare_impl<GreaterThanOp>(mat, threshold, out);
}

std::unique_ptr<Mat> ge(const OriginMat &mat, const OriginMat &threshold, OriginMat *out)
{
    return compare_impl<GreaterEqualOp>(mat, threshold, out);
}

}  // namespace cuda
}  // namespace origin
