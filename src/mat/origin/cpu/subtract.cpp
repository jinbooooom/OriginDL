#include <stdexcept>
#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> subtract(const OriginMat &a, const OriginMat &b)
{
    // 检查数据类型是否匹配
    if (a.dtype() != b.dtype())
    {
        THROW_INVALID_ARG("Data type mismatch for subtraction: expected {} but got {}", dtype_to_string(a.dtype()),
                          dtype_to_string(b.dtype()));
    }

    // 计算广播形状
    Shape result_shape = compute_broadcast_shape(a, b);
    auto result        = std::make_unique<OriginMat>(result_shape, a.dtype());

    // 使用类型分发器执行减法操作
    TypeDispatcher::dispatch_void(
        a.dtype(), [&]<typename T>() { BroadcastCompute::binary_broadcast<T>(a, b, *result, SubtractOp{}); });

    return result;
}

}  // namespace cpu
}  // namespace origin
