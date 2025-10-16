#include <cstring>
#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> reshape(const OriginMat &mat, const Shape &new_shape)
{
    if (new_shape.elements() != mat.elements())
    {
        THROW_INVALID_ARG("Reshape: total elements must match. Original: {}, Target: {}", mat.elements(),
                          new_shape.elements());
    }
    // 创建一个新的OriginMat，共享存储但使用新的形状
    auto result = std::make_unique<OriginMat>(new_shape, mat.dtype());
    // 使用类型分发器复制数据
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        memcpy(result->data_ptr<T>(), mat.data_ptr<T>(), mat.elements() * sizeof(T));
    });
    return result;
}

}  // namespace cpu
}  // namespace origin
