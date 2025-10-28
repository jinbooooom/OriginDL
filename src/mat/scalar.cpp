#include "origin/mat/scalar.h"
#include <sstream>
#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

// 宏定义用于简化类型转换函数的实现
#define SCALAR_CONVERSION_MACRO(return_type, member_name, cast_expr, bool_expr) \
    auto Scalar::to_##member_name() const -> return_type \
    { \
        switch (type_) \
        { \
            case DataType::kFloat32: \
                return cast_expr(v.f32); \
            case DataType::kFloat64: \
                return cast_expr(v.f64); \
            case DataType::kInt8: \
                return cast_expr(v.i8); \
            case DataType::kInt16: \
                return cast_expr(v.i16); \
            case DataType::kInt32: \
                return cast_expr(v.i32); \
            case DataType::kInt64: \
                return cast_expr(v.i64); \
            case DataType::kUInt8: \
                return cast_expr(v.u8); \
            case DataType::kUInt16: \
                return cast_expr(v.u16); \
            case DataType::kUInt32: \
                return cast_expr(v.u32); \
            case DataType::kUInt64: \
                return cast_expr(v.u64); \
            case DataType::kBool: \
                return bool_expr; \
            default: \
                THROW_INVALID_ARG("Cannot convert Scalar to " #return_type " from type: {}", dtype_to_string(type_)); \
        } \
    }

namespace origin
{

// 类型转换实现 - 使用宏简化
SCALAR_CONVERSION_MACRO(float, float32, static_cast<float>, v.b ? 1.0F : 0.0F)
SCALAR_CONVERSION_MACRO(double, float64, static_cast<double>, v.b ? 1.0 : 0.0)
SCALAR_CONVERSION_MACRO(int8_t, int8, static_cast<int8_t>, v.b ? 1 : 0)
SCALAR_CONVERSION_MACRO(int16_t, int16, static_cast<int16_t>, v.b ? 1 : 0)
SCALAR_CONVERSION_MACRO(int32_t, int32, static_cast<int32_t>, v.b ? 1 : 0)
SCALAR_CONVERSION_MACRO(int64_t, int64, static_cast<int64_t>, v.b ? 1 : 0)
SCALAR_CONVERSION_MACRO(uint8_t, uint8, static_cast<uint8_t>, v.b ? 1 : 0)
SCALAR_CONVERSION_MACRO(uint16_t, uint16, static_cast<uint16_t>, v.b ? 1 : 0)
SCALAR_CONVERSION_MACRO(uint32_t, uint32, static_cast<uint32_t>, v.b ? 1 : 0)
SCALAR_CONVERSION_MACRO(uint64_t, uint64, static_cast<uint64_t>, v.b ? 1 : 0)

// bool 转换需要特殊处理
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

// 取消宏定义，避免污染其他文件
#undef SCALAR_CONVERSION_MACRO
