#include <algorithm>
#include <climits>
#include <string>
#include <vector>
#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace utils
{

// 支持的设备类型常量
static const std::vector<std::string> SUPPORTED_DEVICE_TYPES = {"cpu", "cuda"};

// 支持的数据类型常量
static const std::vector<std::string> SUPPORTED_DTYPE_STRINGS = {"float32", "float64", "int32", "int8"};

/**
 * @brief 解析设备字符串，支持PyTorch风格的设备字符串
 * @param device_str 设备字符串，如 "cpu", "cuda", "cuda:0", "cuda:1" 等
 * @return Device 解析后的设备对象
 * @throws std::invalid_argument 当设备字符串格式不正确时
 */
Device parse_device_string(const std::string &device_str)
{
    // 去除前后空格
    std::string trimmed = device_str;
    trimmed.erase(0, trimmed.find_first_not_of(" \t"));
    trimmed.erase(trimmed.find_last_not_of(" \t") + 1);

    if (trimmed.empty())
    {
        THROW_INVALID_ARG("Empty device string provided. Expected one of: cpu, cuda, cuda:0, cuda:1, etc.");
    }

    // 检查CPU设备
    if (trimmed == "cpu")
    {
        return Device(DeviceType::kCPU);
    }

    // 检查CUDA设备
    if (trimmed.substr(0, 4) == "cuda")
    {
        if (trimmed == "cuda")
        {
            return Device(DeviceType::kCUDA, 0);  // 默认GPU 0
        }
        else if (trimmed.length() > 5 && trimmed.substr(0, 5) == "cuda:")
        {
            std::string index_str = trimmed.substr(5);

            // 检查索引部分是否为空
            if (index_str.empty())
            {
                THROW_INVALID_ARG(
                    "Invalid CUDA device string: '{}'. Expected format: 'cuda:0', 'cuda:1', etc. Got: '{}' (missing "
                    "device index after ':')",
                    device_str, device_str);
            }

            // 检查索引部分是否只包含数字
            if (index_str.find_first_not_of("0123456789") != std::string::npos)
            {
                THROW_INVALID_ARG("Invalid CUDA device index: '{}'. Expected non-negative integer, got: '{}'",
                                  index_str, index_str);
            }

            try
            {
                int index = std::stoi(index_str);
                if (index < 0)
                {
                    THROW_INVALID_ARG("Invalid CUDA device index: {}. Expected non-negative integer, got: {}", index,
                                      index);
                }
                return Device(DeviceType::kCUDA, index);
            }
            catch (const std::out_of_range &)
            {
                THROW_INVALID_ARG("CUDA device index out of range: '{}'. Expected integer in range [0, {}]", index_str,
                                  INT_MAX);
            }
        }
        else
        {
            // 类似 'cudax:0' 的错误
            THROW_INVALID_ARG(
                "Invalid device type: '{}'. Expected one of: cpu, cuda, cuda:0, cuda:1, etc. Got: '{}' (unknown device "
                "type)",
                trimmed, device_str);
        }
    }

    // 检查其他可能的设备类型（为未来扩展预留）

    // 生成详细的错误信息
    std::string error_msg = "Invalid device string: '" + device_str + "'. Expected one of: ";
    for (size_t i = 0; i < SUPPORTED_DEVICE_TYPES.size(); ++i)
    {
        error_msg += "'" + SUPPORTED_DEVICE_TYPES[i] + "'";
        if (i < SUPPORTED_DEVICE_TYPES.size() - 1)
        {
            error_msg += ", ";
        }
    }
    error_msg += " or '" + SUPPORTED_DEVICE_TYPES[0] + ":0', '" + SUPPORTED_DEVICE_TYPES[1] + ":1', etc. Got: '" +
                 device_str + "'";

    THROW_INVALID_ARG("{}", error_msg);
}

/**
 * @brief 解析数据类型字符串，支持PyTorch风格的数据类型字符串
 * @param dtype_str 数据类型字符串，如 "float32", "float64", "int32", "int8" 等
 * @return DataType 解析后的数据类型
 * @throws std::invalid_argument 当数据类型字符串格式不正确时
 */
DataType parse_dtype_string(const std::string &dtype_str)
{
    // 去除前后空格并转换为小写
    std::string trimmed = dtype_str;
    trimmed.erase(0, trimmed.find_first_not_of(" \t"));
    trimmed.erase(trimmed.find_last_not_of(" \t") + 1);

    // 转换为小写
    std::transform(trimmed.begin(), trimmed.end(), trimmed.begin(), ::tolower);

    if (trimmed.empty())
    {
        std::string error_msg = "Empty dtype string provided. Expected one of: ";
        for (size_t i = 0; i < SUPPORTED_DTYPE_STRINGS.size(); ++i)
        {
            error_msg += SUPPORTED_DTYPE_STRINGS[i];
            if (i < SUPPORTED_DTYPE_STRINGS.size() - 1)
            {
                error_msg += ", ";
            }
        }
        THROW_INVALID_ARG("{}", error_msg);
    }

    // 支持的数据类型映射
    if (trimmed == "float32" || trimmed == "float")
    {
        return DataType::kFloat32;
    }
    else if (trimmed == "float64" || trimmed == "double")
    {
        return DataType::kFloat64;
    }
    else if (trimmed == "int32" || trimmed == "int")
    {
        return DataType::kInt32;
    }
    else if (trimmed == "int8")
    {
        return DataType::kInt8;
    }
    else
    {
        THROW_INVALID_ARG("Invalid dtype string: '{}'. Expected one of: float32, float64, int32, int8. Got: '{}'",
                          dtype_str, dtype_str);
    }
}

}  // namespace utils
}  // namespace origin
