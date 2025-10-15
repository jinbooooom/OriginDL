#include <cmath>
#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> log(const OriginMat &mat)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *a_data = mat.data_ptr<float>();
            float *c_data       = result->data_ptr<float>();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = std::log(a_data[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = mat.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = std::log(a_data[i]);
            }
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for log operation", dtype_to_string(mat.dtype()));
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
