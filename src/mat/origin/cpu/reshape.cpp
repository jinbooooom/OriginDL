#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include <cstring>
#include <stdexcept>

namespace origin {
namespace cpu {

std::unique_ptr<OriginMat> reshape(const OriginMat& mat, const Shape& new_shape) {
    if (new_shape.elements() != mat.elements()) {
        throw std::invalid_argument("Reshape: total elements must match");
    }
    // 创建一个新的OriginMat，共享存储但使用新的形状
    auto result = std::make_unique<OriginMat>(new_shape, mat.dtype());
    // 复制数据
    switch (mat.dtype()) {
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
            throw std::invalid_argument("Unsupported data type for reshape");
    }
    return result;
}

} // namespace cpu
} // namespace origin
