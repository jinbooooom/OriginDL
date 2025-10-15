#include <algorithm>
#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

data_t min_all(const OriginMat &mat)
{
    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *data = mat.data_ptr<float>();
            float min_val     = data[0];
            for (size_t i = 1; i < mat.elements(); ++i)
            {
                min_val = std::min(min_val, data[i]);
            }
            return static_cast<data_t>(min_val);
        }
        case DataType::kFloat64:
        {
            const double *data = mat.data_ptr<double>();
            double min_val     = data[0];
            for (size_t i = 1; i < mat.elements(); ++i)
            {
                min_val = std::min(min_val, data[i]);
            }
            return static_cast<data_t>(min_val);
        }
        case DataType::kInt32:
        {
            const int32_t *data = mat.data_ptr<int32_t>();
            int32_t min_val     = data[0];
            for (size_t i = 1; i < mat.elements(); ++i)
            {
                min_val = std::min(min_val, data[i]);
            }
            return static_cast<data_t>(min_val);
        }
        case DataType::kInt8:
        {
            const int8_t *data = mat.data_ptr<int8_t>();
            int8_t min_val     = data[0];
            for (size_t i = 1; i < mat.elements(); ++i)
            {
                min_val = std::min(min_val, data[i]);
            }
            return static_cast<data_t>(min_val);
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for min_all operation", dtype_to_string(mat.dtype()));
    }
}

}  // namespace cpu
}  // namespace origin
