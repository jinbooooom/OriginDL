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
 * @brief CPU Sigmoid 激活函数统一实现
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> sigmoid(const OriginMat &mat, OriginMat *out)
{
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute Sigmoid of empty matrix");
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

    const void *a_data = mat.storage()->data();
    void *c_data      = result_ptr->storage()->data();

    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        cpu_unary_kernel<T, SigmoidOp>(static_cast<const T *>(a_data), static_cast<T *>(c_data), mat.elements(),
                                       SigmoidOp{});
    });

    return result_unique;
}

/**
 * @brief CPU Sigmoid 反向传播：gx = gy * y * (1 - y)
 * @param gy 输出梯度
 * @param y 前向传播保存的 sigmoid(x)
 * @return 输入梯度 gx
 */
std::unique_ptr<Mat> sigmoid_backward(const OriginMat &gy, const OriginMat &y)
{
    if (unlikely(gy.elements() == 0) || unlikely(y.elements() != gy.elements()))
    {
        THROW_INVALID_ARG("sigmoid_backward: gy and y must have same non-zero size");
    }
    VALIDATE_CPU_DEVICE(gy);
    VALIDATE_CPU_DEVICE(y);
    if (unlikely(gy.shape() != y.shape() || gy.dtype() != y.dtype()))
    {
        THROW_INVALID_ARG("sigmoid_backward: gy and y must have same shape and dtype");
    }

    auto result = std::make_unique<OriginMat>(gy.shape(), gy.dtype(), gy.device());
    const void *gy_data = gy.storage()->data();
    const void *y_data  = y.storage()->data();
    void *gx_data      = result->storage()->data();
    const size_t n     = gy.elements();

    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        const T *gy_ptr = static_cast<const T *>(gy_data);
        const T *y_ptr  = static_cast<const T *>(y_data);
        T *gx_ptr       = static_cast<T *>(gx_data);
        for (size_t i = 0; i < n; ++i)
        {
            gx_ptr[i] = gy_ptr[i] * y_ptr[i] * (T(1) - y_ptr[i]);
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
