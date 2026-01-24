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

// 数据类型枚举（面向 DL/LLM，所以有些类型不支持）
enum class DataType
{
    kFloat32 = 0,  // float
    kFloat64 = 1,  // double
    kInt8    = 2,  // int8_t
    kInt32   = 3,  // int32_t
    kInt64   = 4,  // int64_t
    kUInt8   = 5,  // uint8_t
};

// 数据类型别名
constexpr auto Float32 = DataType::kFloat32;
constexpr auto Float64 = DataType::kFloat64;
constexpr auto Int8    = DataType::kInt8;
constexpr auto Int32   = DataType::kInt32;
constexpr auto Int64   = DataType::kInt64;
constexpr auto UInt8   = DataType::kUInt8;

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
    static constexpr DataType type    = DataType::kFloat64;
    static constexpr const char *name = "double";
};

template <>
struct DataTypeTraits<int8_t>
{
    static constexpr DataType type    = DataType::kInt8;
    static constexpr const char *name = "int8";
};

template <>
struct DataTypeTraits<int32_t>
{
    static constexpr DataType type    = DataType::kInt32;
    static constexpr const char *name = "int32";
};

template <>
struct DataTypeTraits<int64_t>
{
    static constexpr DataType type    = DataType::kInt64;
    static constexpr const char *name = "int64";
};

template <>
struct DataTypeTraits<uint8_t>
{
    static constexpr DataType type    = DataType::kUInt8;
    static constexpr const char *name = "uint8";
};

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
        case DataType::kInt8:
            return sizeof(int8_t);
        case DataType::kInt32:
            return sizeof(int32_t);
        case DataType::kInt64:
            return sizeof(int64_t);
        case DataType::kUInt8:
            return sizeof(uint8_t);
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

// 编译时字符串获取（推荐使用）
template <typename T>
inline constexpr const char *dtype_to_string()
{
    return DataTypeTraits<T>::name;
}

// 运行时字符串获取（用于错误消息等）
inline std::string dtype_to_string(DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            return DataTypeTraits<float>::name;
        case DataType::kFloat64:
            return DataTypeTraits<double>::name;
        case DataType::kInt8:
            return DataTypeTraits<int8_t>::name;
        case DataType::kInt32:
            return DataTypeTraits<int32_t>::name;
        case DataType::kInt64:
            return DataTypeTraits<int64_t>::name;
        case DataType::kUInt8:
            return DataTypeTraits<uint8_t>::name;
        default:
            return "unknown data type";
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
