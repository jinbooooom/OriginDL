#include <cmath>
#include <stdexcept>
#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> sqrt(const OriginMat &mat)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    // 使用类型分发器执行平方根操作
    TypeDispatcher::dispatch_void(mat.dtype(),
                                  [&]<typename T>() { BroadcastCompute::unary<T>(mat, *result, SqrtOp{}); });

    return result;
}

}  // namespace cpu
}  // namespace origin
