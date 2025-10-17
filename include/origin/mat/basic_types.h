#ifndef __ORIGIN_DL_BASIC_TYPES_H__
#define __ORIGIN_DL_BASIC_TYPES_H__

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace origin
{

// 矩阵计算后端的类型
constexpr int ORIGIN_BACKEND_TYPE = 0;
constexpr int TORCH_BACKEND_TYPE  = 1;

// 数据类型枚举
enum class DataType
{
    kFloat32 = 0,   // float
    kFloat64 = 1,   // double
    kDouble  = 1,   // double
    kInt32   = 2,   // int32_t
    kInt8    = 3,   // int8_t
    kInt16   = 4,   // int16_t
    kInt64   = 5,   // int64_t
    kUInt8   = 6,   // uint8_t
    kUInt16  = 7,   // uint16_t
    kUInt32  = 8,   // uint32_t
    kUInt64  = 9,   // uint64_t
    kBool    = 10,  // bool
};

// 基础数据类型定义（保持向后兼容）
using data_t = float;  // TODO: 未来去掉这个类型

// 数据类型别名，提供更简洁的命名，这样用户可以通过 origin::Float32 去创建 origin::DataType::kFloat32 类型的张量
constexpr auto Float32 = DataType::kFloat32;
constexpr auto Float64 = DataType::kDouble;
constexpr auto Double  = DataType::kDouble;
constexpr auto Int32   = DataType::kInt32;
constexpr auto Int8    = DataType::kInt8;

// 新增类型别名
constexpr auto Int16  = DataType::kInt16;
constexpr auto Int64  = DataType::kInt64;
constexpr auto UInt8  = DataType::kUInt8;
constexpr auto UInt16 = DataType::kUInt16;
constexpr auto UInt32 = DataType::kUInt32;
constexpr auto UInt64 = DataType::kUInt64;
constexpr auto Bool   = DataType::kBool;

/**
 * @brief 获取数据类型的字节大小
 * @param dtype 数据类型
 * @return 该数据类型占用的字节数
 */
inline size_t element_size(DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            return sizeof(float);
        case DataType::kFloat64:
            return sizeof(double);
        case DataType::kInt32:
            return sizeof(int32_t);
        case DataType::kInt8:
            return sizeof(int8_t);
        case DataType::kInt16:
            return sizeof(int16_t);
        case DataType::kInt64:
            return sizeof(int64_t);
        case DataType::kUInt8:
            return sizeof(uint8_t);
        case DataType::kUInt16:
            return sizeof(uint16_t);
        case DataType::kUInt32:
            return sizeof(uint32_t);
        case DataType::kUInt64:
            return sizeof(uint64_t);
        case DataType::kBool:
            return sizeof(bool);
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

/**
 * @brief 类型提升规则
 * @param a 第一个数据类型
 * @param b 第二个数据类型
 * @return 提升后的数据类型
 * @note 优先级：double > float > int64 > int32 > int16 > int8
 */
inline DataType promote_types(DataType a, DataType b)
{
    // 如果类型相同，直接返回
    if (a == b)
        return a;

    // 浮点数优先级最高
    if (a == DataType::kFloat64 || b == DataType::kFloat64)
        return DataType::kFloat64;
    if (a == DataType::kFloat32 || b == DataType::kFloat32)
        return DataType::kFloat32;

    // 整数类型按精度排序
    if (a == DataType::kInt64 || b == DataType::kInt64)
        return DataType::kInt64;
    if (a == DataType::kInt32 || b == DataType::kInt32)
        return DataType::kInt32;
    if (a == DataType::kInt16 || b == DataType::kInt16)
        return DataType::kInt16;
    if (a == DataType::kInt8 || b == DataType::kInt8)
        return DataType::kInt8;

    // 无符号整数
    if (a == DataType::kUInt64 || b == DataType::kUInt64)
        return DataType::kUInt64;
    if (a == DataType::kUInt32 || b == DataType::kUInt32)
        return DataType::kUInt32;
    if (a == DataType::kUInt16 || b == DataType::kUInt16)
        return DataType::kUInt16;
    if (a == DataType::kUInt8 || b == DataType::kUInt8)
        return DataType::kUInt8;

    // 布尔类型
    if (a == DataType::kBool || b == DataType::kBool)
        return DataType::kBool;

    // 默认返回第一个类型
    return a;
}

inline std::string dtype_to_string(DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            return "float32";
        case DataType::kFloat64:
            return "float64";
        case DataType::kInt32:
            return "int32";
        case DataType::kInt8:
            return "int8";
        case DataType::kInt16:
            return "int16";
        case DataType::kInt64:
            return "int64";
        case DataType::kUInt8:
            return "uint8";
        case DataType::kUInt16:
            return "uint16";
        case DataType::kUInt32:
            return "uint32";
        case DataType::kUInt64:
            return "uint64";
        case DataType::kBool:
            return "bool";
        default:
            return "unknown data type";
    }
}

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

// 类型推断内联函数
template <typename T>
inline DataType get_data_type_from_template()
{
    if (std::is_same<T, float>::value)
    {
        return DataType::kFloat32;
    }
    else if (std::is_same<T, double>::value)
    {
        return DataType::kDouble;
    }
    else if (std::is_same<T, int32_t>::value)
    {
        return DataType::kInt32;
    }
    else if (std::is_same<T, int8_t>::value)
    {
        return DataType::kInt8;
    }
    else if (std::is_same<T, unsigned long>::value || std::is_same<T, size_t>::value)
    {
        // 暂时还不支持unsigned long/size_t，先将其映射到int32
        return DataType::kInt32;
    }
    else
    {
        throw std::invalid_argument("Unsupported data type");
    }
}

// 设备类型枚举
enum class DeviceType
{
    kCPU  = 0,
    kCUDA = 1
};

// 设备类
class Device
{
public:
    Device(DeviceType type, int index = 0) : type_(type), index_(index) {}

    DeviceType type() const { return type_; }
    int index() const { return index_; }

    bool operator==(const Device &other) const { return type_ == other.type_ && index_ == other.index_; }

    bool operator!=(const Device &other) const { return !(*this == other); }

    std::string to_string() const
    {
        if (type_ == DeviceType::kCPU)
        {
            return "cpu";
        }
        else if (type_ == DeviceType::kCUDA)
        {
            return "cuda:" + std::to_string(index_);
        }
        return "unknown device type";
    }

private:
    DeviceType type_;
    int index_;
};

// 设备常量
const DeviceType kCPU  = DeviceType::kCPU;
const DeviceType kCUDA = DeviceType::kCUDA;

// 前向声明
namespace utils
{
Device parse_device_string(const std::string &device_str);
DataType parse_dtype_string(const std::string &dtype_str);
}  // namespace utils

// 字符串解析函数 - 委托给utils实现
inline Device parse_device_string(const std::string &device_str)
{
    return utils::parse_device_string(device_str);
}

inline DataType parse_dtype_string(const std::string &dtype_str)
{
    return utils::parse_dtype_string(dtype_str);
}

}  // namespace origin

#endif  // __ORIGIN_DL_BASIC_TYPES_H__
