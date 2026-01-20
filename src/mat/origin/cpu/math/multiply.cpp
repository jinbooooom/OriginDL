#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_kernels.h"
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU乘法算子统一实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> multiply(const OriginMat &a, const OriginMat &b, OriginMat *out)
{
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CPU_DEVICE(a, b);

    Shape result_shape = origin::utils::compute::compute_broadcast_shape(a, b);

    OriginMat *result_ptr = nullptr;
    std::unique_ptr<OriginMat> result_unique;

    if (out != nullptr)
    {
        if (unlikely(out->shape() != result_shape || out->dtype() != a.dtype() || out->device() != a.device()))
        {
            THROW_INVALID_ARG("Output tensor mismatch. Expected shape={}, dtype={}, device={}, but got shape={}, "
                              "dtype={}, device={}",
                              result_shape.to_string(), dtype_to_string(a.dtype()), a.device().to_string(),
                              out->shape().to_string(), dtype_to_string(out->dtype()), out->device().to_string());
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
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            cpu_elementwise_kernel<T, MultiplyOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                                                  static_cast<T *>(c_data), a.elements(), MultiplyOp{});
        });
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            cpu_simple_broadcast_kernel<T, MultiplyOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                                                       static_cast<T *>(c_data), a.elements(), b.elements(),
                                                       result_ptr->elements(), MultiplyOp{});
        });
    }
    else
    {
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CPU multiply operation");
    }

    return result_unique;
}

}  // namespace cpu
}  // namespace origin