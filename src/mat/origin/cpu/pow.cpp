#include <cmath>
#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/scalar.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> pow(const OriginMat &mat, const Scalar &exponent)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *a_data = mat.data_ptr<float>();
            float *c_data       = result->data_ptr<float>();
            float exp           = exponent.to_float32();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = std::pow(a_data[i], exp);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = mat.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();
            double exp           = exponent.to_float64();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = std::pow(a_data[i], exp);
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *a_data = mat.data_ptr<int32_t>();
            int32_t *c_data       = result->data_ptr<int32_t>();
            int32_t exp           = exponent.to_int32();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = static_cast<int32_t>(std::pow(static_cast<double>(a_data[i]), static_cast<double>(exp)));
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *a_data = mat.data_ptr<int8_t>();
            int8_t *c_data       = result->data_ptr<int8_t>();
            int8_t exp           = exponent.to_int8();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = static_cast<int8_t>(std::pow(static_cast<double>(a_data[i]), static_cast<double>(exp)));
            }
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for power operation", dtype_to_string(mat.dtype()));
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
