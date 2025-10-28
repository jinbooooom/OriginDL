#include <stdexcept>
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

// 通用标量操作模板
template <typename Op>
std::unique_ptr<OriginMat> scalar_operation(const OriginMat &mat, data_t scalar, Op op)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        const T *a_data = mat.data_ptr<T>();
        T *c_data       = result->data_ptr<T>();
        T s             = static_cast<T>(scalar);
        for (size_t i = 0; likely(i < mat.elements()); ++i)
        {
            c_data[i] = op(a_data[i], s);
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
