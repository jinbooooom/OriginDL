#include "origin/mat/scalar.h"
#include <sstream>
#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

namespace origin
{

// 类型转换实现
auto Scalar::to_float32() const -> float
{
    switch (type_)
    {
        case DataType::kFloat32:
            return v.f32;
        case DataType::kFloat64:
            return static_cast<float>(v.f64);
        case DataType::kInt8:
            return static_cast<float>(v.i8);
        case DataType::kInt16:
            return static_cast<float>(v.i16);
        case DataType::kInt32:
            return static_cast<float>(v.i32);
        case DataType::kInt64:
            return static_cast<float>(v.i64);
        case DataType::kUInt8:
            return static_cast<float>(v.u8);
        case DataType::kUInt16:
            return static_cast<float>(v.u16);
        case DataType::kUInt32:
            return static_cast<float>(v.u32);
        case DataType::kUInt64:
            return static_cast<float>(v.u64);
        case DataType::kBool:
            return v.b ? 1.0F : 0.0F;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to float from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_float64() const -> double
{
    switch (type_)
    {
        case DataType::kFloat32:
            return static_cast<double>(v.f32);
        case DataType::kFloat64:
            return v.f64;
        case DataType::kInt8:
            return static_cast<double>(v.i8);
        case DataType::kInt16:
            return static_cast<double>(v.i16);
        case DataType::kInt32:
            return static_cast<double>(v.i32);
        case DataType::kInt64:
            return static_cast<double>(v.i64);
        case DataType::kUInt8:
            return static_cast<double>(v.u8);
        case DataType::kUInt16:
            return static_cast<double>(v.u16);
        case DataType::kUInt32:
            return static_cast<double>(v.u32);
        case DataType::kUInt64:
            return static_cast<double>(v.u64);
        case DataType::kBool:
            return v.b ? 1.0 : 0.0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to double from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_int8() const -> int8_t
{
    switch (type_)
    {
        case DataType::kInt8:
            return v.i8;
        case DataType::kInt16:
            return static_cast<int8_t>(v.i16);
        case DataType::kInt32:
            return static_cast<int8_t>(v.i32);
        case DataType::kInt64:
            return static_cast<int8_t>(v.i64);
        case DataType::kUInt8:
            return static_cast<int8_t>(v.u8);
        case DataType::kUInt16:
            return static_cast<int8_t>(v.u16);
        case DataType::kUInt32:
            return static_cast<int8_t>(v.u32);
        case DataType::kUInt64:
            return static_cast<int8_t>(v.u64);
        case DataType::kFloat32:
            return static_cast<int8_t>(v.f32);
        case DataType::kFloat64:
            return static_cast<int8_t>(v.f64);
        case DataType::kBool:
            return v.b ? 1 : 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to int8 from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_int16() const -> int16_t
{
    switch (type_)
    {
        case DataType::kInt16:
            return v.i16;
        case DataType::kInt8:
            return static_cast<int16_t>(v.i8);
        case DataType::kInt32:
            return static_cast<int16_t>(v.i32);
        case DataType::kInt64:
            return static_cast<int16_t>(v.i64);
        case DataType::kUInt8:
            return static_cast<int16_t>(v.u8);
        case DataType::kUInt16:
            return static_cast<int16_t>(v.u16);
        case DataType::kUInt32:
            return static_cast<int16_t>(v.u32);
        case DataType::kUInt64:
            return static_cast<int16_t>(v.u64);
        case DataType::kFloat32:
            return static_cast<int16_t>(v.f32);
        case DataType::kFloat64:
            return static_cast<int16_t>(v.f64);
        case DataType::kBool:
            return v.b ? 1 : 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to int16 from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_int32() const -> int32_t
{
    switch (type_)
    {
        case DataType::kInt32:
            return v.i32;
        case DataType::kInt8:
            return static_cast<int32_t>(v.i8);
        case DataType::kInt16:
            return static_cast<int32_t>(v.i16);
        case DataType::kInt64:
            return static_cast<int32_t>(v.i64);
        case DataType::kUInt8:
            return static_cast<int32_t>(v.u8);
        case DataType::kUInt16:
            return static_cast<int32_t>(v.u16);
        case DataType::kUInt32:
            return static_cast<int32_t>(v.u32);
        case DataType::kUInt64:
            return static_cast<int32_t>(v.u64);
        case DataType::kFloat32:
            return static_cast<int32_t>(v.f32);
        case DataType::kFloat64:
            return static_cast<int32_t>(v.f64);
        case DataType::kBool:
            return v.b ? 1 : 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to int32 from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_int64() const -> int64_t
{
    switch (type_)
    {
        case DataType::kInt64:
            return v.i64;
        case DataType::kInt8:
            return static_cast<int64_t>(v.i8);
        case DataType::kInt16:
            return static_cast<int64_t>(v.i16);
        case DataType::kInt32:
            return static_cast<int64_t>(v.i32);
        case DataType::kUInt8:
            return static_cast<int64_t>(v.u8);
        case DataType::kUInt16:
            return static_cast<int64_t>(v.u16);
        case DataType::kUInt32:
            return static_cast<int64_t>(v.u32);
        case DataType::kUInt64:
            return static_cast<int64_t>(v.u64);
        case DataType::kFloat32:
            return static_cast<int64_t>(v.f32);
        case DataType::kFloat64:
            return static_cast<int64_t>(v.f64);
        case DataType::kBool:
            return v.b ? 1 : 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to int64 from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_uint8() const -> uint8_t
{
    switch (type_)
    {
        case DataType::kUInt8:
            return v.u8;
        case DataType::kUInt16:
            return static_cast<uint8_t>(v.u16);
        case DataType::kUInt32:
            return static_cast<uint8_t>(v.u32);
        case DataType::kUInt64:
            return static_cast<uint8_t>(v.u64);
        case DataType::kInt8:
            return static_cast<uint8_t>(v.i8);
        case DataType::kInt16:
            return static_cast<uint8_t>(v.i16);
        case DataType::kInt32:
            return static_cast<uint8_t>(v.i32);
        case DataType::kInt64:
            return static_cast<uint8_t>(v.i64);
        case DataType::kFloat32:
            return static_cast<uint8_t>(v.f32);
        case DataType::kFloat64:
            return static_cast<uint8_t>(v.f64);
        case DataType::kBool:
            return v.b ? 1 : 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to uint8 from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_uint16() const -> uint16_t
{
    switch (type_)
    {
        case DataType::kUInt16:
            return v.u16;
        case DataType::kUInt8:
            return static_cast<uint16_t>(v.u8);
        case DataType::kUInt32:
            return static_cast<uint16_t>(v.u32);
        case DataType::kUInt64:
            return static_cast<uint16_t>(v.u64);
        case DataType::kInt8:
            return static_cast<uint16_t>(v.i8);
        case DataType::kInt16:
            return static_cast<uint16_t>(v.i16);
        case DataType::kInt32:
            return static_cast<uint16_t>(v.i32);
        case DataType::kInt64:
            return static_cast<uint16_t>(v.i64);
        case DataType::kFloat32:
            return static_cast<uint16_t>(v.f32);
        case DataType::kFloat64:
            return static_cast<uint16_t>(v.f64);
        case DataType::kBool:
            return v.b ? 1 : 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to uint16 from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_uint32() const -> uint32_t
{
    switch (type_)
    {
        case DataType::kUInt32:
            return v.u32;
        case DataType::kUInt8:
            return static_cast<uint32_t>(v.u8);
        case DataType::kUInt16:
            return static_cast<uint32_t>(v.u16);
        case DataType::kUInt64:
            return static_cast<uint32_t>(v.u64);
        case DataType::kInt8:
            return static_cast<uint32_t>(v.i8);
        case DataType::kInt16:
            return static_cast<uint32_t>(v.i16);
        case DataType::kInt32:
            return static_cast<uint32_t>(v.i32);
        case DataType::kInt64:
            return static_cast<uint32_t>(v.i64);
        case DataType::kFloat32:
            return static_cast<uint32_t>(v.f32);
        case DataType::kFloat64:
            return static_cast<uint32_t>(v.f64);
        case DataType::kBool:
            return v.b ? 1 : 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to uint32 from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_uint64() const -> uint64_t
{
    switch (type_)
    {
        case DataType::kUInt64:
            return v.u64;
        case DataType::kUInt8:
            return static_cast<uint64_t>(v.u8);
        case DataType::kUInt16:
            return static_cast<uint64_t>(v.u16);
        case DataType::kUInt32:
            return static_cast<uint64_t>(v.u32);
        case DataType::kInt8:
            return static_cast<uint64_t>(v.i8);
        case DataType::kInt16:
            return static_cast<uint64_t>(v.i16);
        case DataType::kInt32:
            return static_cast<uint64_t>(v.i32);
        case DataType::kInt64:
            return static_cast<uint64_t>(v.i64);
        case DataType::kFloat32:
            return static_cast<uint64_t>(v.f32);
        case DataType::kFloat64:
            return static_cast<uint64_t>(v.f64);
        case DataType::kBool:
            return v.b ? 1 : 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to uint64 from type: {}", dtype_to_string(type_));
    }
}

auto Scalar::to_bool() const -> bool
{
    switch (type_)
    {
        case DataType::kBool:
            return v.b;
        case DataType::kFloat32:
            return v.f32 != 0.0F;
        case DataType::kFloat64:
            return v.f64 != 0.0;
        case DataType::kInt8:
            return v.i8 != 0;
        case DataType::kInt16:
            return v.i16 != 0;
        case DataType::kInt32:
            return v.i32 != 0;
        case DataType::kInt64:
            return v.i64 != 0;
        case DataType::kUInt8:
            return v.u8 != 0;
        case DataType::kUInt16:
            return v.u16 != 0;
        case DataType::kUInt32:
            return v.u32 != 0;
        case DataType::kUInt64:
            return v.u64 != 0;
        default:
            THROW_INVALID_ARG("Cannot convert Scalar to bool from type: {}", dtype_to_string(type_));
    }
}

// 字符串表示
auto Scalar::to_string() const -> std::string
{
    std::ostringstream oss;
    switch (type_)
    {
        case DataType::kFloat32:
            oss << v.f32 << "f";
            break;
        case DataType::kFloat64:
            oss << v.f64;
            break;
        case DataType::kInt8:
            oss << static_cast<int>(v.i8);
            break;
        case DataType::kInt16:
            oss << v.i16;
            break;
        case DataType::kInt32:
            oss << v.i32;
            break;
        case DataType::kInt64:
            oss << v.i64 << "L";
            break;
        case DataType::kUInt8:
            oss << static_cast<unsigned>(v.u8) << "U";
            break;
        case DataType::kUInt16:
            oss << v.u16 << "U";
            break;
        case DataType::kUInt32:
            oss << v.u32 << "U";
            break;
        case DataType::kUInt64:
            oss << v.u64 << "UL";
            break;
        case DataType::kBool:
            oss << (v.b ? "true" : "false");
            break;
        default:
            oss << "unknown scalar type";
            break;
    }
    return oss.str();
}

}  // namespace origin
