#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<Mat> silu(const OriginMat &mat, OriginMat *out)
{
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute SiLU of empty matrix");
    }
    VALIDATE_CPU_DEVICE(mat);

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

    const void *x_data = mat.storage()->data();
    void *y_data       = result_ptr->storage()->data();
    const size_t n     = mat.elements();

    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        const T *x_ptr = static_cast<const T *>(x_data);
        T *y_ptr       = static_cast<T *>(y_data);
        for (size_t i = 0; i < n; ++i)
        {
            const T v = x_ptr[i];
            const T s = T(1) / (T(1) + std::exp(-v));
            y_ptr[i]  = v * s;
        }
    });

    return result_unique;
}

std::unique_ptr<Mat> silu_backward(const OriginMat &gy, const OriginMat &x)
{
    if (unlikely(gy.elements() == 0) || unlikely(x.elements() != gy.elements()))
    {
        THROW_INVALID_ARG("silu_backward: gy and x must have same non-zero size");
    }
    VALIDATE_CPU_DEVICE(gy);
    VALIDATE_CPU_DEVICE(x);
    if (unlikely(gy.shape() != x.shape() || gy.dtype() != x.dtype()))
    {
        THROW_INVALID_ARG("silu_backward: gy and x must have same shape and dtype");
    }

    auto result   = std::make_unique<OriginMat>(gy.shape(), gy.dtype(), gy.device());
    const void *gy_data = gy.storage()->data();
    const void *x_data  = x.storage()->data();
    void *gx_data      = result->storage()->data();
    const size_t n     = gy.elements();

    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        const T *gy_ptr = static_cast<const T *>(gy_data);
        const T *x_ptr  = static_cast<const T *>(x_data);
        T *gx_ptr       = static_cast<T *>(gx_data);
        for (size_t i = 0; i < n; ++i)
        {
            const T v = x_ptr[i];
            const T s = T(1) / (T(1) + std::exp(-v));
            const T grad_silu = s + v * s * (T(1) - s);
            gx_ptr[i]         = gy_ptr[i] * grad_silu;
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin

