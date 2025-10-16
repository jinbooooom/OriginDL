#ifndef __ORIGIN_DL_TENSOR_OPTIONS_H__
#define __ORIGIN_DL_TENSOR_OPTIONS_H__

#include "../mat/basic_types.h"

namespace origin
{

/**
 * @brief 张量选项配置类，用于统一管理张量的各种属性，支持链式调用。
*/
class TensorOptions
{
public:
    TensorOptions() = default;

    // 隐式构造函数，支持直接传递单个选项
    explicit TensorOptions(DataType dtype) : dtype_(dtype) {}
    explicit TensorOptions(Device device) : device_(device) {}

    // 链式调用方法
    TensorOptions &dtype(DataType dtype)
    {
        dtype_ = dtype;
        return *this;
    }

    TensorOptions &dtype(const std::string& dtype_str)
    {
        dtype_ = parse_dtype_string(dtype_str);
        return *this;
    }

    TensorOptions &dtype(const char* dtype_str)
    {
        return dtype(std::string(dtype_str));
    }

    TensorOptions &device(Device device)
    {
        device_ = device;
        return *this;
    }

    // 支持DeviceType + index
    TensorOptions &device(DeviceType device_type, int index = 0)
    {
        device_ = Device(device_type, index);
        return *this;
    }

    TensorOptions &device(const std::string& device_str)
    {
        device_ = parse_device_string(device_str);
        return *this;
    }

    TensorOptions &device(const char* device_str)
    {
        return device(std::string(device_str));
    }

    TensorOptions &requires_grad(bool requires_grad)
    {
        requires_grad_ = requires_grad;
        return *this;
    }

    // 访问器
    DataType dtype() const { return dtype_; }
    Device device() const { return device_; }
    bool requires_grad() const { return requires_grad_; }

    // 比较操作
    bool operator==(const TensorOptions &other) const
    {
        return dtype_ == other.dtype_ && device_ == other.device_ && requires_grad_ == other.requires_grad_;
    }

    bool operator!=(const TensorOptions &other) const { return !(*this == other); }

    // 转换为字符串（调试用）
    std::string to_string() const
    {
        std::string result = "TensorOptions(dtype=";
        result += std::to_string(static_cast<int>(dtype_));
        result += ", device=";
        result += device_.to_string();
        result += ", requires_grad=";
        result += (requires_grad_ ? "true" : "false");
        result += ")";
        return result;
    }

private:
    DataType dtype_ = DataType::kFloat32;
    Device device_  = Device(DeviceType::kCPU);
    bool requires_grad_ = true;  // TODO：当前的origindl不支持requires_grad=false，所以默认是true，未来支持后，需要修改
};

inline TensorOptions dtype(DataType dtype)
{
    return TensorOptions(dtype);
}

inline TensorOptions dtype(const std::string& dtype_str)
{
    return TensorOptions().dtype(dtype_str);
}

inline TensorOptions dtype(const char* dtype_str)
{
    return TensorOptions().dtype(dtype_str);
}

inline TensorOptions device(Device device)
{
    return TensorOptions().device(device);
}

inline TensorOptions device(DeviceType device_type, int index = 0)
{
    return TensorOptions().device(device_type, index);
}

inline TensorOptions device(const std::string& device_str)
{
    return TensorOptions().device(device_str);
}

inline TensorOptions device(const char* device_str)
{
    return TensorOptions().device(device_str);
}

inline TensorOptions requires_grad(bool requires_grad = true)
{
    return TensorOptions().requires_grad(requires_grad);
}

}  // namespace origin

#endif  // __ORIGIN_DL_TENSOR_OPTIONS_H__
