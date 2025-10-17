#ifndef __ORIGIN_DL_SCALAR_H__
#define __ORIGIN_DL_SCALAR_H__

#include <string>
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
 * - int16_t (DataType::kInt16)
 * - int32_t (DataType::kInt32)
 * - int64_t (DataType::kInt64)
 * - uint8_t (DataType::kUInt8)
 * - uint16_t (DataType::kUInt16)
 * - uint32_t (DataType::kUInt32)
 * - uint64_t (DataType::kUInt64)
 * - bool (DataType::kBool)
 *
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
        int8_t i8;    // kInt8
        int16_t i16;  // kInt16
        int32_t i32;  // kInt32
        int64_t i64;  // kInt64

        // 无符号整数类型
        uint8_t u8;    // kUInt8
        uint16_t u16;  // kUInt16
        uint32_t u32;  // kUInt32
        uint64_t u64;  // kUInt64

        // 布尔类型
        bool b;  // kBool
    } v;
    DataType type_;  // 使用OriginDL现有的DataType

public:
    // 构造函数 - 支持所有类型
    Scalar(float value) : type_(DataType::kFloat32) { v.f32 = value; }
    Scalar(double value) : type_(DataType::kFloat64) { v.f64 = value; }
    Scalar(int8_t value) : type_(DataType::kInt8) { v.i8 = value; }
    Scalar(int16_t value) : type_(DataType::kInt16) { v.i16 = value; }
    Scalar(int32_t value) : type_(DataType::kInt32) { v.i32 = value; }
    Scalar(int64_t value) : type_(DataType::kInt64) { v.i64 = value; }
    Scalar(uint8_t value) : type_(DataType::kUInt8) { v.u8 = value; }
    Scalar(uint16_t value) : type_(DataType::kUInt16) { v.u16 = value; }
    Scalar(uint32_t value) : type_(DataType::kUInt32) { v.u32 = value; }
    Scalar(uint64_t value) : type_(DataType::kUInt64) { v.u64 = value; }
    Scalar(bool value) : type_(DataType::kBool) { v.b = value; }

    // 默认构造函数
    Scalar() : type_(DataType::kFloat32) { v.f32 = 0.0F; }

    // 类型转换 - 支持所有类型
    [[nodiscard]] auto to_float32() const -> float;
    [[nodiscard]] auto to_float64() const -> double;
    [[nodiscard]] auto to_int8() const -> int8_t;
    [[nodiscard]] auto to_int16() const -> int16_t;
    [[nodiscard]] auto to_int32() const -> int32_t;
    [[nodiscard]] auto to_int64() const -> int64_t;
    [[nodiscard]] auto to_uint8() const -> uint8_t;
    [[nodiscard]] auto to_uint16() const -> uint16_t;
    [[nodiscard]] auto to_uint32() const -> uint32_t;
    [[nodiscard]] auto to_uint64() const -> uint64_t;
    [[nodiscard]] auto to_bool() const -> bool;

    // 转换为data_t（保持向后兼容）
    [[nodiscard]] auto toDataT() const -> data_t { return to_float32(); }  // data_t是float的别名

    // 类型查询
    [[nodiscard]] auto dtype() const -> DataType { return type_; }

    [[nodiscard]] auto is_floating_point() const -> bool
    {
        return type_ == DataType::kFloat32 || type_ == DataType::kFloat64;
    }

    [[nodiscard]] auto is_integral() const -> bool
    {
        return type_ == DataType::kInt8 || type_ == DataType::kInt16 || type_ == DataType::kInt32 ||
               type_ == DataType::kInt64 || type_ == DataType::kUInt8 || type_ == DataType::kUInt16 ||
               type_ == DataType::kUInt32 || type_ == DataType::kUInt64;
    }

    [[nodiscard]] auto is_boolean() const -> bool { return type_ == DataType::kBool; }

    [[nodiscard]] auto is_signed() const -> bool
    {
        return type_ == DataType::kInt8 || type_ == DataType::kInt16 || type_ == DataType::kInt32 ||
               type_ == DataType::kInt64;
    }

    [[nodiscard]] auto is_unsigned() const -> bool
    {
        return type_ == DataType::kUInt8 || type_ == DataType::kUInt16 || type_ == DataType::kUInt32 ||
               type_ == DataType::kUInt64;
    }

    // 字符串表示
    [[nodiscard]] auto to_string() const -> std::string;
};

}  // namespace origin

#endif  // __ORIGIN_DL_SCALAR_H__
