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
    kInt8    = 2,   // int8_t
    kInt16   = 3,   // int16_t
    kInt32   = 4,   // int32_t
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
constexpr auto Int8    = DataType::kInt8;
constexpr auto Int16   = DataType::kInt16;
constexpr auto Int32   = DataType::kInt32;
constexpr auto Int64   = DataType::kInt64;
constexpr auto UInt8   = DataType::kUInt8;
constexpr auto UInt16  = DataType::kUInt16;
constexpr auto UInt32  = DataType::kUInt32;
constexpr auto UInt64  = DataType::kUInt64;
constexpr auto Bool    = DataType::kBool;

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
struct DataTypeTraits<int8_t>
{
    static constexpr DataType type    = DataType::kInt8;
    static constexpr const char *name = "int8";
};

template <>
struct DataTypeTraits<int16_t>
{
    static constexpr DataType type    = DataType::kInt16;
    static constexpr const char *name = "int16";
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

template <>
struct DataTypeTraits<uint16_t>
{
    static constexpr DataType type    = DataType::kUInt16;
    static constexpr const char *name = "uint16";
};

template <>
struct DataTypeTraits<uint32_t>
{
    static constexpr DataType type    = DataType::kUInt32;
    static constexpr const char *name = "uint32";
};

template <>
struct DataTypeTraits<uint64_t>
{
    static constexpr DataType type    = DataType::kUInt64;
    static constexpr const char *name = "uint64";
};

template <>
struct DataTypeTraits<bool>
{
    static constexpr DataType type    = DataType::kBool;
    static constexpr const char *name = "bool";
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
        case DataType::kInt16:
            return sizeof(int16_t);
        case DataType::kInt32:
            return sizeof(int32_t);
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
        case DataType::kInt16:
            return DataTypeTraits<int16_t>::name;
        case DataType::kInt32:
            return DataTypeTraits<int32_t>::name;
        case DataType::kInt64:
            return DataTypeTraits<int64_t>::name;
        case DataType::kUInt8:
            return DataTypeTraits<uint8_t>::name;
        case DataType::kUInt16:
            return DataTypeTraits<uint16_t>::name;
        case DataType::kUInt32:
            return DataTypeTraits<uint32_t>::name;
        case DataType::kUInt64:
            return DataTypeTraits<uint64_t>::name;
        case DataType::kBool:
            return DataTypeTraits<bool>::name;
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
