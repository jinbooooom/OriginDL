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
 * - int32_t (DataType::kInt32)
 * - int8_t (DataType::kInt8)
 * 
 */
class Scalar
{
private:
    // 使用对齐的union确保内存安全
    union alignas(8) {
        float f;       // 单精度浮点数
        double d;      // 双精度浮点数
        int32_t i;     // 32位整数
        int8_t c;      // 8位整数
    } v;
    DataType type_;    // 使用OriginDL现有的DataType
    
public:
    // 构造函数 - 支持隐式转换
    Scalar(float value) : type_(DataType::kFloat32) { v.f = value; }
    Scalar(double value) : type_(DataType::kFloat64) { v.d = value; }
    Scalar(int32_t value) : type_(DataType::kInt32) { v.i = value; }
    Scalar(int8_t value) : type_(DataType::kInt8) { v.c = value; }
    
    // 默认构造函数
    Scalar() : type_(DataType::kFloat32) { v.f = 0.0F; }
    
    // 类型转换 - 带错误检查
    [[nodiscard]] auto to_float32() const -> float;
    [[nodiscard]] auto to_float64() const -> double;
    [[nodiscard]] auto to_int32() const -> int32_t;
    [[nodiscard]] auto to_int8() const -> int8_t;
    
    // 转换为data_t（保持向后兼容）
    [[nodiscard]] auto toDataT() const -> data_t { return to_float32(); }  // data_t是float的别名
    
    // 类型查询
    [[nodiscard]] auto dtype() const -> DataType { return type_; }
    
    [[nodiscard]] auto is_floating_point() const -> bool {
        return type_ == DataType::kFloat32 || type_ == DataType::kFloat64;
    }
    
    [[nodiscard]] auto is_integral() const -> bool {
        return type_ == DataType::kInt32 || type_ == DataType::kInt8;
    }
    
    // 字符串表示
    [[nodiscard]] auto to_string() const -> std::string;
};

}  // namespace origin

#endif  // __ORIGIN_DL_SCALAR_H__
