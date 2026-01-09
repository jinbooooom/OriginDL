#include <cuda_runtime.h>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
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
 * @brief divide算子实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 除法结果矩阵
 */
std::unique_ptr<Mat> divide(const OriginMat &a, const OriginMat &b)
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
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            launch_elementwise_kernel<T, DivideOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                                                   static_cast<T *>(c_data), a.elements(), DivideOp{}, 0);
        });
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        // 简单广播：一个操作数是标量（次常见）
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            launch_simple_broadcast_kernel<T, DivideOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                                                        static_cast<T *>(c_data), a.elements(), b.elements(),
                                                        result->elements(), DivideOp{}, 0);
        });
    }
    else
    {
        // 复杂广播：需要计算步长信息
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CUDA divide operation");
    }

    CUDA_CHECK_ASYNC();

    return result;
}

/**
 * @brief CUDA原地除法算子实现（将目标矩阵除以源矩阵）
 * @param a 目标矩阵（会被修改）
 * @param b 源矩阵（不会被修改）
 * @note 要求 a 和 b 的形状相同
 */
void divide_inplace(OriginMat &a, const OriginMat &b)
{
    // 验证输入
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CUDA_DEVICE(a, b);

    // 验证形状必须相同（原地操作要求）
    if (a.shape() != b.shape())
    {
        THROW_INVALID_ARG("divide_inplace: shapes must match. a.shape() = {}, b.shape() = {}", a.shape().to_string(),
                          b.shape().to_string());
    }

    // 获取数据指针
    void *a_data       = a.storage()->data();
    const void *b_data = b.storage()->data();

    // 执行原地除法：a = a / b
    device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
        launch_elementwise_kernel<T, DivideOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                                               static_cast<T *>(a_data), a.elements(), DivideOp{}, 0);
    });

    CUDA_CHECK_ASYNC();
}

}  // namespace cuda
}  // namespace origin
