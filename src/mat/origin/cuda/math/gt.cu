#include <cuda_runtime.h>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief 广播大于kernel - threshold是标量
 * @details A和C是长度为N的向量，threshold是标量（长度为1），计算 C[i] = (A[i] > threshold[0]) ? 1 : 0
 */
template <typename T>
__global__ void gt_broadcast_kernel(const T *__restrict__ A, const T *__restrict__ threshold, T *__restrict__ C, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        GreaterThanOp op;
        C[i] = op(A[i], threshold[0]);
    }
}

/**
 * @brief 向量化广播大于kernel - float4版本，threshold是标量
 * @details A和C是长度为N的向量，threshold是标量（长度为1），使用float4向量化优化
 *          计算 C[i:i+4] = (A[i:i+4] > threshold[0]) ? 1 : 0
 */
__global__ void gt_broadcast_vectorized_float4_kernel(const float *__restrict__ A,
                                                      const float *__restrict__ threshold,
                                                      float *__restrict__ C,
                                                      size_t N)
{
    constexpr size_t VECTOR_SIZE = 4;
    size_t vectorized_N          = (N / VECTOR_SIZE) * VECTOR_SIZE;
    size_t vector_idx            = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

    GreaterThanOp op;
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

/**
 * @brief CUDA大于算子统一实现（对标量）
 * @param mat 输入矩阵
 * @param threshold 标量阈值
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> gt(const OriginMat &mat, const Scalar &threshold, OriginMat *out)
{
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute gt of empty matrix");
    }
    VALIDATE_CUDA_DEVICE(mat);

    // 创建标量tensor用于广播比较
    Shape scalar_shape{};  // 标量形状
    TensorOptions scalar_options = TensorOptions().dtype(mat.dtype()).device(mat.device());
    auto threshold_mat_unique = OriginMat::from_scalar(threshold, scalar_shape, scalar_options);
    const OriginMat &threshold_mat = static_cast<const OriginMat &>(*threshold_mat_unique);

    Shape result_shape = mat.shape();  // 结果形状与输入相同

    OriginMat *result_ptr = nullptr;
    std::unique_ptr<OriginMat> result_unique;

    if (out != nullptr)
    {
        if (unlikely(out->shape() != result_shape || out->dtype() != mat.dtype() || out->device() != mat.device()))
        {
            THROW_INVALID_ARG(
                "Output tensor mismatch. Expected shape={}, dtype={}, device={}, but got shape={}, "
                "dtype={}, device={}",
                result_shape.to_string(), dtype_to_string(mat.dtype()), mat.device().to_string(), out->shape().to_string(),
                dtype_to_string(out->dtype()), out->device().to_string());
        }
        result_ptr = out;
    }
    else
    {
        result_unique = std::make_unique<OriginMat>(result_shape, mat.dtype(), mat.device());
        result_ptr    = result_unique.get();
    }

    const void *mat_data      = mat.storage()->data();
    const void *threshold_data = threshold_mat.storage()->data();
    void *result_data         = result_ptr->storage()->data();

    // 标量广播：使用广播kernel
    const size_t num_elements = result_ptr->elements();

    // 仅对float类型使用向量化优化，其他类型使用基础版本
    if (mat.dtype() == DataType::kFloat32)
    {
        constexpr size_t VECTOR_SIZE     = 4;
        const size_t threads_per_block   = 256;
        const size_t vectorized_elements = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
        const size_t num_blocks          = (vectorized_elements + threads_per_block - 1) / threads_per_block;
        gt_broadcast_vectorized_float4_kernel<<<num_blocks, threads_per_block>>>(
            static_cast<const float *>(mat_data), static_cast<const float *>(threshold_data),
            static_cast<float *>(result_data), num_elements);
    }
    else
    {
        const size_t threads_per_block = 256;
        const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            gt_broadcast_kernel<T><<<num_blocks, threads_per_block>>>(static_cast<const T *>(mat_data),
                                                                      static_cast<const T *>(threshold_data),
                                                                      static_cast<T *>(result_data), num_elements);
        });
    }

    return result_unique;
}

}  // namespace cuda
}  // namespace origin
