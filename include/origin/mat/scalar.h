#ifndef __ORIGIN_DL_SCALAR_H__
#define __ORIGIN_DL_SCALAR_H__

#include <string>
#include <type_traits>
#include "basic_types.h"

namespace origin
{

/**
 * @brief 标量值类，支持多种数值类型
 * @details 使用union存储不同类型的标量值，提供类型安全的标量操作
 *
 * 支持的类型：
 * - float (DataType::kFloat32)
 * - double (DataType::kFloat64)
 * - int8_t (DataType::kInt8)
 * - int32_t (DataType::kInt32)
 * - int64_t (DataType::kInt64)
 * - uint8_t (DataType::kUInt8)
 */
class Scalar
{
private:
    // 使用对齐的union确保内存安全
    union alignas(8)
    {
        // 浮点数类型
        float f32;   // kFloat32
        double f64;  // kFloat64/kDouble

        // 有符号整数类型
        int8_t i8;   // kInt8
        int32_t i32;  // kInt32
        int64_t i64;  // kInt64

        // 无符号整数类型
        uint8_t u8;  // kUInt8
    } v;
    DataType type_;

public:
    // 构造函数 - 支持所有类型
    Scalar(float value) : type_(DataType::kFloat32) { v.f32 = value; }
    Scalar(double value) : type_(DataType::kFloat64) { v.f64 = value; }
    Scalar(int8_t value) : type_(DataType::kInt8) { v.i8 = value; }
    Scalar(int32_t value) : type_(DataType::kInt32) { v.i32 = value; }
    Scalar(int64_t value) : type_(DataType::kInt64) { v.i64 = value; }
    Scalar(uint8_t value) : type_(DataType::kUInt8) { v.u8 = value; }

    // 默认构造函数
    Scalar() : type_(DataType::kFloat32) { v.f32 = 0.0F; }

    // 类型转换 - 支持所有类型
    [[nodiscard]] auto to_float32() const -> float;
    [[nodiscard]] auto to_float64() const -> double;
    [[nodiscard]] auto to_int8() const -> int8_t;
    [[nodiscard]] auto to_int32() const -> int32_t;
    [[nodiscard]] auto to_int64() const -> int64_t;
    [[nodiscard]] auto to_uint8() const -> uint8_t;

    // 通用模板转换函数 - 使用特化方式以获得最佳性能
    template <typename T>
    [[nodiscard]] auto to() const -> T;

    // 类型查询
    [[nodiscard]] auto dtype() const -> DataType { return type_; }

    [[nodiscard]] auto is_floating_point() const -> bool
    {
        return type_ == DataType::kFloat32 || type_ == DataType::kFloat64;
    }

    [[nodiscard]] auto is_integral() const -> bool
    {
        return type_ == DataType::kInt8 || type_ == DataType::kInt32 || type_ == DataType::kInt64 ||
               type_ == DataType::kUInt8;
    }

    [[nodiscard]] auto is_signed() const -> bool
    {
        return type_ == DataType::kInt8 || type_ == DataType::kInt32 || type_ == DataType::kInt64;
    }

    [[nodiscard]] auto is_unsigned() const -> bool { return type_ == DataType::kUInt8; }

    // 字符串表示
    [[nodiscard]] auto to_string() const -> std::string;
};

// auto to() const -> T 模板特化实现
template <>
inline auto Scalar::to<float>() const -> float
{
    return to_float32();
}

template <>
inline auto Scalar::to<double>() const -> double
{
    return to_float64();
}

template <>
inline auto Scalar::to<int8_t>() const -> int8_t
{
    return to_int8();
}

template <>
inline auto Scalar::to<int32_t>() const -> int32_t
{
    return to_int32();
}

template <>
inline auto Scalar::to<int64_t>() const -> int64_t
{
    return to_int64();
}

template <>
inline auto Scalar::to<uint8_t>() const -> uint8_t
{
    return to_uint8();
}

// 用于 Tensor::item<unsigned long> 等；uint64 已从 DataType 移除，经 int64 转换
template <>
inline auto Scalar::to<unsigned long>() const -> unsigned long
{
    return static_cast<unsigned long>(to_int64());
}

}  // namespace origin

#endif  // __ORIGIN_DL_SCALAR_H__
