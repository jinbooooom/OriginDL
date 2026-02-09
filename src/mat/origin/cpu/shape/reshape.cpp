#include <cstring>
#include <stdexcept>
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU reshape算子实现
 * @deprecated 此函数不会被调用
 * 
 * ============================================================================
 * 重要说明：此函数不会被调用
 * ============================================================================
 * 
 * OriginMat::reshape() 在 origin_mat.cpp 中已经实现了基于视图的 reshape 逻辑：
 * - 如果张量是连续的，使用 view() 创建视图（零拷贝）
 * - 如果张量不是连续的，先创建连续副本，然后对连续副本使用 view()
 * 
 * 因此，此函数永远不会被调用。保留此文件仅用于：
 * - 历史兼容性
 * - 未来可能的特殊用途
 * 
 * 实际的 reshape 逻辑请参考：src/mat/origin/origin_mat.cpp (396-415行)
 * ============================================================================
 */

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
