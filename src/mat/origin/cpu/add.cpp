#include <stdexcept>
#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"
#include "origin/utils/branch_prediction.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> add(const OriginMat &a, const OriginMat &b)
{
    // 检查数据类型是否匹配 - 使用分支预测优化
    if (unlikely(a.dtype() != b.dtype()))
    {
        THROW_INVALID_ARG("Data type mismatch for addition: expected {} but got {}", dtype_to_string(a.dtype()),
                          dtype_to_string(b.dtype()));
    }

    // 计算广播形状
    Shape result_shape = compute_broadcast_shape(a, b);
    auto result        = std::make_unique<OriginMat>(result_shape, a.dtype());

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
    3. BroadcastCompute::binary_broadcast<T>
    这是广播计算模板：
    <T>：使用TypeDispatcher推断出的具体类型
    处理两个矩阵的广播运算（形状匹配、标量广播等）
    支持三种情况：
    相同形状：直接元素级运算
    a是标量：广播到b的形状
    b是标量：广播到a的形状
    */
    device_common::TypeDispatcher::dispatch_void(a.dtype(),
                                  [&]<typename T>() { BroadcastCompute::binary_broadcast<T>(a, b, *result, AddOp{}); });

    return result;
}

}  // namespace cpu
}  // namespace origin
