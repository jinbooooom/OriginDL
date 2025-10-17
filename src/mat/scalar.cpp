#include "origin/mat/scalar.h"
#include "origin/utils/exception.h"
#include "origin/mat/basic_types.h"
#include <sstream>

namespace origin
{

// 类型转换实现
auto Scalar::to_float32() const -> float
{
    switch (type_)
    {
        case DataType::kFloat32:
            return v.f;
        case DataType::kFloat64:
            return static_cast<float>(v.d);
        case DataType::kInt32:
            return static_cast<float>(v.i);
        case DataType::kInt8:
            return static_cast<float>(v.c);
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to float from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_float64() const -> double
{
    switch (type_)
    {
        case DataType::kFloat32:
            return static_cast<double>(v.f);
        case DataType::kFloat64:
            return v.d;
        case DataType::kInt32:
            return static_cast<double>(v.i);
        case DataType::kInt8:
            return static_cast<double>(v.c);
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to double from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_int32() const -> int32_t
{
    switch (type_)
    {
        case DataType::kFloat32:
            return static_cast<int32_t>(v.f);
        case DataType::kFloat64:
            return static_cast<int32_t>(v.d);
        case DataType::kInt32:
            return v.i;
        case DataType::kInt8:
            return static_cast<int32_t>(v.c);
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to int32 from type: {}", static_cast<int>(type_));
    }
}

auto Scalar::to_int8() const -> int8_t
{
    switch (type_)
    {
        case DataType::kFloat32:
            return static_cast<int8_t>(v.f);
        case DataType::kFloat64:
            return static_cast<int8_t>(v.d);
        case DataType::kInt32:
            return static_cast<int8_t>(v.i);
        case DataType::kInt8:
            return v.c;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to int8 from type: {}", static_cast<int>(type_));
    }
}

// 字符串表示
auto Scalar::to_string() const -> std::string
{
    std::ostringstream oss;
    switch (type_)
    {
        case DataType::kFloat32:
            oss << v.f << "f";
            break;
        case DataType::kFloat64:
            oss << v.d;
            break;
        case DataType::kInt32:
            oss << v.i;
            break;
        case DataType::kInt8:
            oss << static_cast<int>(v.c);
            break;
        default:
            oss << "unknown scalar type";
            break;
    }
    return oss.str();
}

}  // namespace origin
