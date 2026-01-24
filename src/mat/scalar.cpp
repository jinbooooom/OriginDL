#include "origin/mat/scalar.h"
#include <sstream>
#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

// 宏定义用于简化类型转换函数的实现
#define SCALAR_CONVERSION_MACRO(return_type, member_name, cast_expr)                                                  \
    auto Scalar::to_##member_name() const->return_type                                                                \
    {                                                                                                                 \
        switch (type_)                                                                                                \
        {                                                                                                             \
            case DataType::kFloat32:                                                                                  \
                return cast_expr(v.f32);                                                                              \
            case DataType::kFloat64:                                                                                  \
                return cast_expr(v.f64);                                                                              \
            case DataType::kInt8:                                                                                     \
                return cast_expr(v.i8);                                                                               \
            case DataType::kInt32:                                                                                    \
                return cast_expr(v.i32);                                                                              \
            case DataType::kInt64:                                                                                    \
                return cast_expr(v.i64);                                                                              \
            case DataType::kUInt8:                                                                                    \
                return cast_expr(v.u8);                                                                               \
            default:                                                                                                  \
                THROW_INVALID_ARG("Cannot convert Scalar to " #return_type " from type: {}", dtype_to_string(type_)); \
        }                                                                                                             \
    }

namespace origin
{

// 类型转换实现 - 使用宏简化
SCALAR_CONVERSION_MACRO(float, float32, static_cast<float>)
SCALAR_CONVERSION_MACRO(double, float64, static_cast<double>)
SCALAR_CONVERSION_MACRO(int8_t, int8, static_cast<int8_t>)
SCALAR_CONVERSION_MACRO(int32_t, int32, static_cast<int32_t>)
SCALAR_CONVERSION_MACRO(int64_t, int64, static_cast<int64_t>)
SCALAR_CONVERSION_MACRO(uint8_t, uint8, static_cast<uint8_t>)

// 字符串表示
// 类型后缀说明：
// - float:    使用 "f" 后缀 (如 3.14f)
// - double:   无后缀，默认浮点类型 (如 3.14)
// - int8_t:   无后缀，C++标准中无对应字面量后缀 (如 42)
// - int32_t:  无后缀，C++标准中无对应字面量后缀 (如 42)
// - int64_t:  使用 "LL" 后缀，对应 long long (如 42LL)
// - uint8_t:  使用 "U" 后缀，转换为 unsigned 显示 (如 42U)
// 注意：int8_t/int32_t 在C++标准中没有对应的字面量后缀，
//       因此直接显示数值，必要时进行类型转换以避免符号扩展问题
auto Scalar::to_string() const -> std::string
{
    std::ostringstream oss;
    switch (type_)
    {
        case DataType::kFloat32:
            oss << v.f32 << "f";
            break;
        case DataType::kFloat64:
            oss << v.f64;  // double 默认无后缀
            break;
        case DataType::kInt8:
            oss << static_cast<int>(v.i8);
            break;
        case DataType::kInt32:
            oss << v.i32;
            break;
        case DataType::kInt64:
            oss << v.i64 << "LL";
            break;
        case DataType::kUInt8:
            oss << static_cast<unsigned>(v.u8) << "U";
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
