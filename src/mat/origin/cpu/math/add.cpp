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
 * @brief CPU加法算子实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 加法结果矩阵
 */
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b)
{
    // 输入验证
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CPU_DEVICE(a, b);

    // 计算广播形状
    Shape result_shape = origin::utils::compute::compute_broadcast_shape(a, b);
    auto result        = std::make_unique<OriginMat>(result_shape, a.dtype(), a.device());

    // 获取数据指针
    const void *a_data = a.storage()->data();
    const void *b_data = b.storage()->data();
    void *c_data       = result->storage()->data();

    // 分支优化 - 与CUDA保持一致
    if (a.shape() == b.shape())
    {
        // 相同形状：直接元素级运算
        // 使用类型分发器执行加法操作
        /*
        1. device_common::TypeDispatcher::dispatch_void
        这是一个类型分发器，它的作用是：
        根据 a.dtype() 返回的数据类型（如 kFloat32、kFloat64 等）
        自动选择对应的C++类型（如 float、double 等）
        调用传入的lambda函数，并传递正确的类型参数
        2. Lambda表达式 [&]<typename T>()
        这是C++20的模板lambda语法：
        [&]：捕获所有外部变量 by reference
        <typename T>()：模板参数，T会被TypeDispatcher自动推断
        当TypeDispatcher检测到数据类型是kFloat32时，T就是float
        当检测到kFloat64时，T就是double
        3. cpu_elementwise_kernel<T, AddOp>
        <T>：使用TypeDispatcher推断出的具体类型。
        */
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            cpu_elementwise_kernel<T, AddOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                                             static_cast<T *>(c_data), a.elements(), AddOp{});
        });
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        // 简单广播：标量广播
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            cpu_simple_broadcast_kernel<T, AddOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                                                  static_cast<T *>(c_data), a.elements(), b.elements(),
                                                  result->elements(), AddOp{});
        });
    }
    else
    {
        // 复杂广播：需要计算步长信息
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CPU add operation");
    }

    return result;
}

/**
 * @brief CPU原地加法算子实现（累加到目标矩阵）
 * @param a 目标矩阵（会被修改）
 * @param b 源矩阵（不会被修改）
 * @note 要求 a 和 b 的形状相同
 */
void add_inplace(OriginMat &a, const OriginMat &b)
{
    // 输入验证
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CPU_DEVICE(a, b);

    // 验证形状必须相同（原地操作要求）
    if (a.shape() != b.shape())
    {
        THROW_INVALID_ARG("add_inplace: shapes must match. a.shape() = {}, b.shape() = {}", a.shape().to_string(),
                          b.shape().to_string());
    }

    // 获取数据指针
    void *a_data       = a.storage()->data();
    const void *b_data = b.storage()->data();

    // 执行原地加法：a = a + b
    device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
        cpu_elementwise_kernel<T, AddOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data),
                                         static_cast<T *>(a_data), a.elements(), AddOp{});
    });
}

}  // namespace cpu
}  // namespace origin
