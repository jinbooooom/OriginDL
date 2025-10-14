#include <cmath>
#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> exp(const OriginMat &mat)
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
                c_data[i] = std::exp(a_data[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = mat.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = std::exp(a_data[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for exp");
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
