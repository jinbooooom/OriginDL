#ifndef __ORIGIN_DL_BASIC_TYPES_H__
#define __ORIGIN_DL_BASIC_TYPES_H__

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <vector>
#include <climits>

namespace origin
{

// 矩阵计算后端的类型
constexpr int ORIGIN_BACKEND_TYPE = 0;
constexpr int TORCH_BACKEND_TYPE  = 1;

// 数据类型枚举
enum class DataType
{
    kFloat32 = 0,  // float
    kFloat64 = 1,  // double
    kDouble  = 1,  // double
    kInt32   = 2,  // int32_t
    kInt8    = 3   // int8_t
};

// 基础数据类型定义（保持向后兼容）
using data_t = float;  // TODO: 未来去掉这个类型

// 数据类型别名，提供更简洁的命名，这样用户可以通过 origin::Float32 去创建 origin::DataType::kFloat32 类型的张量
constexpr auto Float32 = DataType::kFloat32;
constexpr auto Float64 = DataType::kDouble;
constexpr auto Double  = DataType::kDouble;
constexpr auto Int32   = DataType::kInt32;
constexpr auto Int8    = DataType::kInt8;

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
namespace utils {
    Device parse_device_string(const std::string& device_str);
    DataType parse_dtype_string(const std::string& dtype_str);
}

// 字符串解析函数 - 委托给utils实现
inline Device parse_device_string(const std::string& device_str) {
    return utils::parse_device_string(device_str);
}

inline DataType parse_dtype_string(const std::string& dtype_str) {
    return utils::parse_dtype_string(dtype_str);
}

}  // namespace origin

#endif  // __ORIGIN_DL_BASIC_TYPES_H__

