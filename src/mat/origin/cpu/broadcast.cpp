#include <stdexcept>
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> broadcast_to(const OriginMat &mat, const Shape &target_shape)
{
    // 简单的实现：创建目标形状的矩阵并复制数据
    auto result = std::make_unique<OriginMat>(target_shape, mat.dtype());

    // 使用类型分发器执行广播操作
    device_common::TypeDispatcher::dispatch_void(
        mat.dtype(), [&]<typename T>() { BroadcastToCompute::broadcast_to<T>(mat, *result); });

    return result;
}

}  // namespace cpu
}  // namespace origin
