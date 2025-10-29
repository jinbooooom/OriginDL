#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_kernels.h"
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU乘法算子实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 乘法结果矩阵
 */
std::unique_ptr<Mat> multiply(const OriginMat &a, const OriginMat &b)
{
    // 输入验证 - 与CUDA保持一致
    if (unlikely(a.dtype() != b.dtype()))
    {
        THROW_INVALID_ARG("Data type mismatch in CPU multiply: {} vs {}", 
                          dtype_to_string(a.dtype()), dtype_to_string(b.dtype()));
    }

    if (unlikely(a.device() != b.device()))
    {
        THROW_INVALID_ARG("Device mismatch in CPU multiply: {} vs {}", 
                          a.device().to_string(), b.device().to_string());
    }

    if (unlikely(a.device().type() != DeviceType::kCPU))
    {
        THROW_INVALID_ARG("Device mismatch in CPU multiply: expected CPU device, got {}", 
                          a.device().to_string());
    }

    // 计算广播形状
    Shape result_shape = origin::utils::compute::compute_broadcast_shape(a, b);
    auto result = std::make_unique<OriginMat>(result_shape, a.dtype(), a.device());

    // 获取数据指针
    const void *a_data = a.storage()->data();
    const void *b_data = b.storage()->data();
    void *c_data = result->storage()->data();

    // 分支优化 - 与CUDA保持一致
    if (a.shape() == b.shape())
    {
        // 相同形状：直接元素级运算
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            cpu_elementwise_kernel<T, MultiplyOp>(static_cast<const T *>(a_data), 
                                                 static_cast<const T *>(b_data), 
                                                 static_cast<T *>(c_data), 
                                                 a.elements(), MultiplyOp{});
        });
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        // 简单广播：标量广播
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            cpu_simple_broadcast_kernel<T, MultiplyOp>(static_cast<const T *>(a_data), 
                                                      static_cast<const T *>(b_data), 
                                                      static_cast<T *>(c_data), 
                                                      a.elements(), b.elements(), 
                                                      result->elements(), MultiplyOp{});
        });
    }
    else
    {
        // 复杂广播：需要计算步长信息
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CPU multiply operation");
    }

    return result;
}

}  // namespace cpu
}  // namespace origin