#include <cstring>
#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
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
    // 复制数据
    switch (mat.dtype())
    {
        case DataType::kFloat32:
            memcpy(result->data_ptr<float>(), mat.data_ptr<float>(), mat.elements() * sizeof(float));
            break;
        case DataType::kFloat64:
            memcpy(result->data_ptr<double>(), mat.data_ptr<double>(), mat.elements() * sizeof(double));
            break;
        case DataType::kInt32:
            memcpy(result->data_ptr<int32_t>(), mat.data_ptr<int32_t>(), mat.elements() * sizeof(int32_t));
            break;
        case DataType::kInt8:
            memcpy(result->data_ptr<int8_t>(), mat.data_ptr<int8_t>(), mat.elements() * sizeof(int8_t));
            break;
        default:
            THROW_INVALID_ARG("Unsupported data type {} for reshape operation", dtype_to_string(mat.dtype()));
    }
    return result;
}

}  // namespace cpu
}  // namespace origin
