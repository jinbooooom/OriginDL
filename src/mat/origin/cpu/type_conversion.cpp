#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> convert_datatype(const OriginMat &mat, DataType target_type)
{
    if (target_type == mat.dtype())
    {
        return std::make_unique<OriginMat>(mat);
    }

    auto result = std::make_unique<OriginMat>(mat.shape(), target_type);

    // 使用双重类型分发执行类型转换，因为有两层对 dtype 做 switch case，所以需要两次分发
    device_common::TypeDispatcher::dispatch_void(/*一次分发*/ mat.dtype(), [&]<typename SrcT>() {
        device_common::TypeDispatcher::dispatch_void(/*二次分发*/ target_type, [&]<typename DstT>() {
            TypeConversionCompute::convert<SrcT, DstT>(mat, *result);
        });
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
