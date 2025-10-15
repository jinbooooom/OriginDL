#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> convert_datatype(const OriginMat &mat, DataType target_type)
{
    if (target_type == mat.dtype())
    {
        return std::make_unique<OriginMat>(mat);
    }

    auto result = std::make_unique<OriginMat>(mat.shape(), target_type);

    // 类型转换
    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *src = mat.data_ptr<float>();
            switch (target_type)
            {
                case DataType::kFloat64:
                {
                    double *dst = result->data_ptr<double>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<double>(src[i]);
                    }
                    break;
                }
                case DataType::kInt32:
                {
                    int32_t *dst = result->data_ptr<int32_t>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<int32_t>(src[i]);
                    }
                    break;
                }
                case DataType::kInt8:
                {
                    int8_t *dst = result->data_ptr<int8_t>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<int8_t>(src[i]);
                    }
                    break;
                }
                default:
                    THROW_INVALID_ARG("Unsupported target type {} for conversion from {}", dtype_to_string(target_type),
                                      dtype_to_string(mat.dtype()));
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *src = mat.data_ptr<double>();
            switch (target_type)
            {
                case DataType::kFloat32:
                {
                    float *dst = result->data_ptr<float>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<float>(src[i]);
                    }
                    break;
                }
                case DataType::kInt32:
                {
                    int32_t *dst = result->data_ptr<int32_t>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<int32_t>(src[i]);
                    }
                    break;
                }
                case DataType::kInt8:
                {
                    int8_t *dst = result->data_ptr<int8_t>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<int8_t>(src[i]);
                    }
                    break;
                }
                default:
                    THROW_INVALID_ARG("Unsupported target type {} for conversion from {}", dtype_to_string(target_type),
                                      dtype_to_string(mat.dtype()));
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *src = mat.data_ptr<int32_t>();
            switch (target_type)
            {
                case DataType::kFloat32:
                {
                    float *dst = result->data_ptr<float>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<float>(src[i]);
                    }
                    break;
                }
                case DataType::kFloat64:
                {
                    double *dst = result->data_ptr<double>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<double>(src[i]);
                    }
                    break;
                }
                case DataType::kInt8:
                {
                    int8_t *dst = result->data_ptr<int8_t>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<int8_t>(src[i]);
                    }
                    break;
                }
                default:
                    THROW_INVALID_ARG("Unsupported target type {} for conversion from {}", dtype_to_string(target_type),
                                      dtype_to_string(mat.dtype()));
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *src = mat.data_ptr<int8_t>();
            switch (target_type)
            {
                case DataType::kFloat32:
                {
                    float *dst = result->data_ptr<float>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<float>(src[i]);
                    }
                    break;
                }
                case DataType::kFloat64:
                {
                    double *dst = result->data_ptr<double>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<double>(src[i]);
                    }
                    break;
                }
                case DataType::kInt32:
                {
                    int32_t *dst = result->data_ptr<int32_t>();
                    for (size_t i = 0; i < mat.shape().elements(); ++i)
                    {
                        dst[i] = static_cast<int32_t>(src[i]);
                    }
                    break;
                }
                default:
                    THROW_INVALID_ARG("Unsupported target type {} for conversion from {}", dtype_to_string(target_type),
                                      dtype_to_string(mat.dtype()));
            }
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported source type {} for conversion", dtype_to_string(mat.dtype()));
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
