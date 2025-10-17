#include <random>
#include <stdexcept>
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

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
            THROW_INVALID_ARG("Unsupported data type {} for randn operation", dtype_to_string(options.dtype()));
    }

    return result;
}

std::unique_ptr<OriginMat> zeros(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        T *data = result->data_ptr<T>();
        for (size_t i = 0; i < shape.elements(); ++i)
        {
            data[i] = static_cast<T>(0);
        }
    });

    return result;
}

std::unique_ptr<OriginMat> ones(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        T *data = result->data_ptr<T>();
        for (size_t i = 0; i < shape.elements(); ++i)
        {
            data[i] = static_cast<T>(1);
        }
    });

    return result;
}

std::unique_ptr<OriginMat> full(const Shape &shape, data_t value, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        T *data = result->data_ptr<T>();
        T v     = static_cast<T>(value);
        for (size_t i = 0; i < shape.elements(); ++i)
        {
            data[i] = v;
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
