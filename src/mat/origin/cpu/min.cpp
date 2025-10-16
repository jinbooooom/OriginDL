#include <algorithm>
#include <stdexcept>
#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

data_t min_all(const OriginMat &mat)
{
    // 使用类型分发器执行最小值计算
    return device_common::TypeDispatcher::dispatch(mat.dtype(), [&]<typename T>() -> data_t {
        T min_val = ReductionCompute::min_all<T>(mat);
        return static_cast<data_t>(min_val);
    });
}

}  // namespace cpu
}  // namespace origin
