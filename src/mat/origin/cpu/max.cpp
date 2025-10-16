#include <algorithm>
#include <stdexcept>
#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

data_t max_all(const OriginMat &mat)
{
    // 使用类型分发器执行最大值计算
    return device_common::TypeDispatcher::dispatch(mat.dtype(), [&]<typename T>() -> data_t {
        T max_val = ReductionCompute::max_all<T>(mat);
        return static_cast<data_t>(max_val);
    });
}

}  // namespace cpu
}  // namespace origin
