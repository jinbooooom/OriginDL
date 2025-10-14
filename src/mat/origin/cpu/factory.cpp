#include <random>
#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> randn(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    std::random_device rd;
    std::mt19937 gen(rd());

    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            float *data = result->data_ptr<float>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = dist(gen);
            }
            break;
        }
        case DataType::kFloat64:
        {
            std::normal_distribution<double> dist(0.0, 1.0);
            double *data = result->data_ptr<double>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = dist(gen);
            }
            break;
        }
        case DataType::kInt32:
        {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            int32_t *data = result->data_ptr<int32_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = static_cast<int32_t>(dist(gen));
            }
            break;
        }
        case DataType::kInt8:
        {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            int8_t *data = result->data_ptr<int8_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = static_cast<int8_t>(dist(gen));
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for randn");
    }

    return result;
}

std::unique_ptr<OriginMat> zeros(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            float *data = result->data_ptr<float>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 0.0f;
            }
            break;
        }
        case DataType::kFloat64:
        {
            double *data = result->data_ptr<double>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 0.0;
            }
            break;
        }
        case DataType::kInt32:
        {
            int32_t *data = result->data_ptr<int32_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 0;
            }
            break;
        }
        case DataType::kInt8:
        {
            int8_t *data = result->data_ptr<int8_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 0;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for zeros");
    }

    return result;
}

std::unique_ptr<OriginMat> ones(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            float *data = result->data_ptr<float>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 1.0f;
            }
            break;
        }
        case DataType::kFloat64:
        {
            double *data = result->data_ptr<double>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 1.0;
            }
            break;
        }
        case DataType::kInt32:
        {
            int32_t *data = result->data_ptr<int32_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 1;
            }
            break;
        }
        case DataType::kInt8:
        {
            int8_t *data = result->data_ptr<int8_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 1;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for ones");
    }

    return result;
}

std::unique_ptr<OriginMat> full(const Shape &shape, data_t value, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            float *data = result->data_ptr<float>();
            float v     = static_cast<float>(value);
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = v;
            }
            break;
        }
        case DataType::kFloat64:
        {
            double *data = result->data_ptr<double>();
            double v     = static_cast<double>(value);
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = v;
            }
            break;
        }
        case DataType::kInt32:
        {
            int32_t *data = result->data_ptr<int32_t>();
            int32_t v     = static_cast<int32_t>(value);
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = v;
            }
            break;
        }
        case DataType::kInt8:
        {
            int8_t *data = result->data_ptr<int8_t>();
            int8_t v     = static_cast<int8_t>(value);
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = v;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for full");
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
