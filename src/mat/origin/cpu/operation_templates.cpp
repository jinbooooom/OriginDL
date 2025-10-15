#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

Shape compute_broadcast_shape(const OriginMat &a, const OriginMat &b)
{
    if (a.shape() == b.shape())
    {
        // 形状相同，直接返回
        return a.shape();
    }
    else if (a.elements() == 1)
    {
        // a是标量，返回b的形状
        return b.shape();
    }
    else if (b.elements() == 1)
    {
        // b是标量，返回a的形状
        return a.shape();
    }
    else
    {
        THROW_INVALID_ARG(
            "Shape mismatch for broadcast operation - only scalar broadcasting supported. Shape A: {}, Shape B: {}",
            a.shape().to_string(), b.shape().to_string());
    }
}

}  // namespace cpu
}  // namespace origin
