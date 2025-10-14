#include "origin/mat/origin/origin_mat.h"
#include <stdexcept>

namespace origin {
namespace cpu {

std::unique_ptr<OriginMat> square(const OriginMat& mat) {
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    switch (mat.dtype()) {
        case DataType::kFloat32: {
            const float* a_data = mat.data_ptr<float>();
            float* c_data = result->data_ptr<float>();
            for (size_t i = 0; i < mat.elements(); ++i) {
                c_data[i] = a_data[i] * a_data[i];
            }
            break;
        }
        case DataType::kFloat64: {
            const double* a_data = mat.data_ptr<double>();
            double* c_data = result->data_ptr<double>();
            for (size_t i = 0; i < mat.elements(); ++i) {
                c_data[i] = a_data[i] * a_data[i];
            }
            break;
        }
        case DataType::kInt32: {
            const int32_t* a_data = mat.data_ptr<int32_t>();
            int32_t* c_data = result->data_ptr<int32_t>();
            for (size_t i = 0; i < mat.elements(); ++i) {
                c_data[i] = a_data[i] * a_data[i];
            }
            break;
        }
        case DataType::kInt8: {
            const int8_t* a_data = mat.data_ptr<int8_t>();
            int8_t* c_data = result->data_ptr<int8_t>();
            for (size_t i = 0; i < mat.elements(); ++i) {
                c_data[i] = a_data[i] * a_data[i];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for square");
    }

    return result;
}

} // namespace cpu
} // namespace origin
