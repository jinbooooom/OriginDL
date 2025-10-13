#ifndef __ORIGIN_DL_BASIC_TYPES_H__
#define __ORIGIN_DL_BASIC_TYPES_H__

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace origin
{

// 数据类型枚举
enum class DataType
{
    kFloat32 = 0,  // float
    kDouble  = 1,  // double
    kInt32   = 2,  // int32_t
    kInt8    = 3   // int8_t
};

// 基础数据类型定义（保持向后兼容）
using data_t = float; // TODO: 未来去掉这个类型

// 数据类型别名，提供更简洁的命名，这样用户可以通过 origin::Float32 去创建 origin::DataType::kFloat32 类型的张量
constexpr auto Float32 = DataType::kFloat32;
constexpr auto Double = DataType::kDouble;
constexpr auto Float64 = DataType::kDouble;
constexpr auto Int32 = DataType::kInt32;
constexpr auto Int8 = DataType::kInt8;

// 类型特征模板，用于在编译时确定数据类型
template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr DataType type    = DataType::kFloat32;
    static constexpr const char *name = "float32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr DataType type    = DataType::kDouble;
    static constexpr const char *name = "double";
};

template <>
struct DataTypeTraits<int32_t>
{
    static constexpr DataType type    = DataType::kInt32;
    static constexpr const char *name = "int32";
};

template <>
struct DataTypeTraits<int8_t>
{
    static constexpr DataType type    = DataType::kInt8;
    static constexpr const char *name = "int8";
};

// 从DataType枚举获取对应的C++类型
template <DataType DT>
struct TypeFromDataType;

template <>
struct TypeFromDataType<DataType::kFloat32>
{
    using type = float;
};

template <>
struct TypeFromDataType<DataType::kDouble>
{
    using type = double;
};

template <>
struct TypeFromDataType<DataType::kInt32>
{
    using type = int32_t;
};

template <>
struct TypeFromDataType<DataType::kInt8>
{
    using type = int8_t;
};

// 类型推断内联函数
template <typename T>
inline DataType get_data_type_from_template()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return DataType::kFloat32;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return DataType::kDouble;
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        return DataType::kInt32;
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
        return DataType::kInt8;
    }
    else if constexpr (std::is_same_v<T, unsigned long> || std::is_same_v<T, size_t>)
    {
        // 将unsigned long/size_t映射到int32，因为通常用于索引和计数
        return DataType::kInt32;
    }
    else
    {
        throw std::invalid_argument("Unsupported data type");
    }
}

// 矩阵计算后端的类型
constexpr int ORIGIN_BACKEND_TYPE = 0;
constexpr int TORCH_BACKEND_TYPE  = 1;

}  // namespace origin

#endif  // __ORIGIN_DL_BASIC_TYPES_H__
