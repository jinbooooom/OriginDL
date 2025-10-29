#include <cuda_runtime.h>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_broadcast.cuh"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"
#include "origin/utils/branch_prediction.h"

namespace origin
{
namespace cuda
{

/**
 * @brief add算子实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 加法结果矩阵
 */
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b)
{
    // 验证输入
    if (unlikely(a.dtype() != b.dtype()))
    {
        THROW_INVALID_ARG("Data type mismatch in CUDA add: {} vs {}", dtype_to_string(a.dtype()),
                          dtype_to_string(b.dtype()));
    }

    if (unlikely(a.device() != b.device()))
    {
        THROW_INVALID_ARG("Device mismatch in CUDA add: {} vs {}", a.device().to_string(), b.device().to_string());
    }

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
            launch_elementwise_kernel<T, AddOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data), 
                                               static_cast<T *>(c_data), a.elements(), AddOp{}, 0);
        });
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        // 简单广播：一个操作数是标量（次常见）
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            launch_simple_broadcast_kernel<T, AddOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data), 
                                                    static_cast<T *>(c_data), a.elements(), b.elements(), 
                                                    result->elements(), AddOp{}, 0);
        });
    }
    else
    {
        // 复杂广播：需要计算步长信息
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CUDA add operation");
    }

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
