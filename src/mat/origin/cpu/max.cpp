#include "origin/mat/origin/origin_mat.h"
#include <algorithm>
#include <stdexcept>

namespace origin {
namespace cpu {

data_t max_all(const OriginMat& mat) {
    switch (mat.dtype()) {
        case DataType::kFloat32: {
            const float* data = mat.data_ptr<float>();
            float max_val = data[0];
            for (size_t i = 1; i < mat.elements(); ++i) {
                max_val = std::max(max_val, data[i]);
            }
            return static_cast<data_t>(max_val);
        }
        case DataType::kFloat64: {
            const double* data = mat.data_ptr<double>();
            double max_val = data[0];
            for (size_t i = 1; i < mat.elements(); ++i) {
                max_val = std::max(max_val, data[i]);
            }
            return static_cast<data_t>(max_val);
        }
        case DataType::kInt32: {
            const int32_t* data = mat.data_ptr<int32_t>();
            int32_t max_val = data[0];
            for (size_t i = 1; i < mat.elements(); ++i) {
                max_val = std::max(max_val, data[i]);
            }
            return static_cast<data_t>(max_val);
        }
        case DataType::kInt8: {
            const int8_t* data = mat.data_ptr<int8_t>();
            int8_t max_val = data[0];
            for (size_t i = 1; i < mat.elements(); ++i) {
                max_val = std::max(max_val, data[i]);
            }
            return static_cast<data_t>(max_val);
        }
        default:
            throw std::invalid_argument("Unsupported data type for max_all");
    }
}

} // namespace cpu
} // namespace origin
