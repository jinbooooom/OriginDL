#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> add_scalar(const OriginMat &mat, data_t scalar)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *a_data = mat.data_ptr<float>();
            float *c_data       = result->data_ptr<float>();
            float s             = static_cast<float>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] + s;
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = mat.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();
            double s             = static_cast<double>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] + s;
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *a_data = mat.data_ptr<int32_t>();
            int32_t *c_data       = result->data_ptr<int32_t>();
            int32_t s             = static_cast<int32_t>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] + s;
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *a_data = mat.data_ptr<int8_t>();
            int8_t *c_data       = result->data_ptr<int8_t>();
            int8_t s             = static_cast<int8_t>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] + s;
            }
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for scalar addition", dtype_to_string(mat.dtype()));
    }

    return result;
}

std::unique_ptr<OriginMat> subtract_scalar(const OriginMat &mat, data_t scalar)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *a_data = mat.data_ptr<float>();
            float *c_data       = result->data_ptr<float>();
            float s             = static_cast<float>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] - s;
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = mat.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();
            double s             = static_cast<double>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] - s;
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *a_data = mat.data_ptr<int32_t>();
            int32_t *c_data       = result->data_ptr<int32_t>();
            int32_t s             = static_cast<int32_t>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] - s;
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *a_data = mat.data_ptr<int8_t>();
            int8_t *c_data       = result->data_ptr<int8_t>();
            int8_t s             = static_cast<int8_t>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] - s;
            }
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for scalar subtraction", dtype_to_string(mat.dtype()));
    }

    return result;
}

std::unique_ptr<OriginMat> multiply_scalar(const OriginMat &mat, data_t scalar)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *a_data = mat.data_ptr<float>();
            float *c_data       = result->data_ptr<float>();
            float s             = static_cast<float>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] * s;
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = mat.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();
            double s             = static_cast<double>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] * s;
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *a_data = mat.data_ptr<int32_t>();
            int32_t *c_data       = result->data_ptr<int32_t>();
            int32_t s             = static_cast<int32_t>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] * s;
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *a_data = mat.data_ptr<int8_t>();
            int8_t *c_data       = result->data_ptr<int8_t>();
            int8_t s             = static_cast<int8_t>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] * s;
            }
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for scalar multiplication", dtype_to_string(mat.dtype()));
    }

    return result;
}

std::unique_ptr<OriginMat> divide_scalar(const OriginMat &mat, data_t scalar)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *a_data = mat.data_ptr<float>();
            float *c_data       = result->data_ptr<float>();
            float s             = static_cast<float>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] / s;
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = mat.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();
            double s             = static_cast<double>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] / s;
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *a_data = mat.data_ptr<int32_t>();
            int32_t *c_data       = result->data_ptr<int32_t>();
            int32_t s             = static_cast<int32_t>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] / s;
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *a_data = mat.data_ptr<int8_t>();
            int8_t *c_data       = result->data_ptr<int8_t>();
            int8_t s             = static_cast<int8_t>(scalar);
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = a_data[i] / s;
            }
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for scalar division", dtype_to_string(mat.dtype()));
    }

    return result;
}

std::unique_ptr<OriginMat> negate(const OriginMat &mat)
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
                c_data[i] = -a_data[i];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = mat.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = -a_data[i];
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *a_data = mat.data_ptr<int32_t>();
            int32_t *c_data       = result->data_ptr<int32_t>();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = -a_data[i];
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *a_data = mat.data_ptr<int8_t>();
            int8_t *c_data       = result->data_ptr<int8_t>();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                c_data[i] = -a_data[i];
            }
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for negation", dtype_to_string(mat.dtype()));
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
