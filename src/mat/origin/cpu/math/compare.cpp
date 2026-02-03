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
 * @brief CPU比较算子统一实现
 * @tparam Op 比较操作类型（EqualOp, NotEqualOp, LessThanOp, LessEqualOp, GreaterThanOp, GreaterEqualOp）
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
    VALIDATE_SAME_CPU_DEVICE(mat, threshold);

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
        THROW_INVALID_ARG("Compare operator: threshold must be scalar (shape {{}} or {{1}}) or have same shape as input. "
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
    const void *threshold_data   = threshold.storage()->data();
    void *result_data           = result_ptr->storage()->data();

    // 分支优化 - 参考 add.cpp 的实现方式
    if (mat.shape() == threshold.shape())
    {
        // 相同形状：直接元素级运算
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            cpu_elementwise_kernel<T, Op>(static_cast<const T *>(mat_data),
                                          static_cast<const T *>(threshold_data),
                                          static_cast<T *>(result_data), mat.elements(), Op{});
        });
    }
    else if (is_scalar_threshold)
    {
        // 标量广播：使用简单广播kernel
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            cpu_simple_broadcast_kernel<T, Op>(static_cast<const T *>(mat_data),
                                               static_cast<const T *>(threshold_data),
                                               static_cast<T *>(result_data), mat.elements(), threshold.elements(),
                                               result_ptr->elements(), Op{});
        });
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

}  // namespace cpu
}  // namespace origin
