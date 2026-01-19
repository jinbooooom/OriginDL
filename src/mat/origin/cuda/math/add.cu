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

/**
 * @brief 元素级加法kernel（相同形状）
 * @details 每个线程处理一个元素的加法运算
 */
template <typename T>
__global__ void add_elementwise_kernel(const T *__restrict__ A,
                                       const T *__restrict__ B,
                                       T *__restrict__ C,
                                       size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief 简单广播加法kernel
 * @details 处理标量广播情况，其中一个操作数是标量（只有1个元素）
 */
template <typename T>
__global__ void add_broadcast_kernel(const T *__restrict__ A,
                                     const T *__restrict__ B,
                                     T *__restrict__ C,
                                     size_t a_elements,
                                     size_t b_elements,
                                     size_t c_elements)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < c_elements)
    {
        size_t a_idx = (a_elements == 1) ? 0 : i;
        size_t b_idx = (b_elements == 1) ? 0 : i;

        C[i] = A[a_idx] + B[b_idx];
    }
}

/**
 * @brief add算子实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 加法结果矩阵
 */
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b)
{
    // 验证输入
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CUDA_DEVICE(a, b);

    // 计算广播形状
    Shape result_shape = origin::utils::compute::compute_broadcast_shape(a, b);
    auto result        = std::make_unique<OriginMat>(result_shape, a.dtype(), a.device());

    // 获取数据指针
    const void *a_data = a.storage()->data();
    const void *b_data = b.storage()->data();
    void *c_data       = result->storage()->data();

    // 检查运算类型，按出现频率排序
    if (a.shape() == b.shape())
    {
        // 相同形状：直接元素级运算（最常见）
        // 使用256线程块，这是CUDA文档推荐的常见选择
        const size_t threads_per_block = 256;
        const size_t num_elements      = a.elements();
        const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;

        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            // 直接使用<<<grid, block>>>语法启动kernel（按照CUDA文档风格）
            add_elementwise_kernel<T><<<num_blocks, threads_per_block>>>(
                static_cast<const T *>(a_data), static_cast<const T *>(b_data), static_cast<T *>(c_data),
                num_elements);
        });
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        // 简单广播：一个操作数是标量（次常见）
        const size_t threads_per_block = 256;
        const size_t num_elements      = result->elements();
        const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;

        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            // 直接使用<<<grid, block>>>语法启动kernel（按照CUDA文档风格）
            add_broadcast_kernel<T><<<num_blocks, threads_per_block>>>(
                static_cast<const T *>(a_data), static_cast<const T *>(b_data), static_cast<T *>(c_data),
                a.elements(), b.elements(), num_elements);
        });
    }
    else
    {
        // 复杂广播：需要计算步长信息
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CUDA add operation");
    };

    return result;
}

/**
 * @brief CUDA原地加法算子实现（累加到目标矩阵）
 * @param a 目标矩阵（会被修改）
 * @param b 源矩阵（不会被修改）
 * @note 要求 a 和 b 的形状相同
 */
void add_inplace(OriginMat &a, const OriginMat &b)
{
    // 验证输入
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CUDA_DEVICE(a, b);

    // 验证形状必须相同（原地操作要求）
    if (a.shape() != b.shape())
    {
        THROW_INVALID_ARG("add_inplace: shapes must match. a.shape() = {}, b.shape() = {}", a.shape().to_string(),
                          b.shape().to_string());
    }

    // 获取数据指针
    void *a_data       = a.storage()->data();
    const void *b_data = b.storage()->data();

    // 使用256线程块，这是CUDA文档推荐的常见选择
    const size_t threads_per_block = 256;
    const size_t num_elements      = a.elements();
    const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;

    // 执行原地加法：a = a + b
    // 直接使用<<<grid, block>>>语法启动kernel（按照CUDA文档风格）
    device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
        add_elementwise_kernel<T><<<num_blocks, threads_per_block>>>(
            static_cast<const T *>(a_data), static_cast<const T *>(b_data), static_cast<T *>(a_data),
            num_elements);
    });

    CUDA_CHECK_ASYNC();
}

}  // namespace cuda
}  // namespace origin
