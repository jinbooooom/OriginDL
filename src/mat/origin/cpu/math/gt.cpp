#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_kernels.h"
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU大于算子统一实现（对标量）
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
    VALIDATE_CPU_DEVICE(mat);

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

    const void *mat_data        = mat.storage()->data();
    const void *threshold_data   = threshold_mat.storage()->data();
    void *result_data           = result_ptr->storage()->data();

    // 标量广播：使用简单广播kernel
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        cpu_simple_broadcast_kernel<T, GreaterThanOp>(static_cast<const T *>(mat_data),
                                                       static_cast<const T *>(threshold_data),
                                                       static_cast<T *>(result_data), mat.elements(), 1,
                                                       result_ptr->elements(), GreaterThanOp{});
    });

    return result_unique;
}

}  // namespace cpu
}  // namespace origin
