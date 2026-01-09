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
 * @brief CPU平方根运算实现
 * @param mat 输入矩阵
 * @return 平方根运算结果矩阵
 */
std::unique_ptr<Mat> sqrt(const OriginMat &mat)
{
    // 输入验证
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute square root of empty matrix");
    }
    VALIDATE_CPU_DEVICE(mat);
    VALIDATE_FLOAT_DTYPE(mat);

    // 创建结果矩阵
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype(), mat.device());

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    // 使用类型分发器执行平方根运算
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        cpu_unary_kernel<T, SqrtOp>(static_cast<const T *>(a_data), static_cast<T *>(c_data), mat.elements(), SqrtOp{});
    });

    return result;
}

/**
 * @brief CPU原地平方根运算实现（修改当前矩阵）
 * @param mat 输入矩阵（会被修改）
 */
void sqrt_inplace(OriginMat &mat)
{
    // 输入验证
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute square root of empty matrix");
    }
    VALIDATE_CPU_DEVICE(mat);
    VALIDATE_FLOAT_DTYPE(mat);

    // 获取数据指针
    void *a_data = mat.storage()->data();

    // 使用类型分发器执行原地平方根运算
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        cpu_unary_kernel<T, SqrtOp>(static_cast<const T *>(a_data), static_cast<T *>(a_data), mat.elements(), SqrtOp{});
    });
}

}  // namespace cpu
}  // namespace origin
