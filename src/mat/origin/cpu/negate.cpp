#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> negate(const OriginMat &mat)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());
    
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        BroadcastCompute::unary<T>(mat, *result, NegOp{});
    });
    
    return result;
}

}  // namespace cpu
}  // namespace origin
