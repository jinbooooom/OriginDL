#ifndef __ORIGIN_DL_EXCEPTION_H__
#define __ORIGIN_DL_EXCEPTION_H__

#include <filesystem>
#include <stdexcept>
#include <string>
#include "spdlog/fmt/fmt.h"

namespace origin
{

// 工具函数：获取文件名（去掉路径）
inline const char *basename(const char *file)
{
    const char *last_slash     = strrchr(file, '/');
    const char *last_backslash = strrchr(file, '\\');

    // 找到最后一个路径分隔符
    const char *last_sep = last_slash;
    if (last_backslash && (!last_slash || last_backslash > last_slash))
    {
        last_sep = last_backslash;
    }

    return last_sep ? last_sep + 1 : file;
}

// 工具函数：获取文件名（去掉路径）
inline std::string basename(const std::string &file)
{
    return std::string(basename(file.c_str()));
}

}  // namespace origin

// 通用THROW宏 - 支持格式化字符串，格式：文件名：函数名：行号
#define THROW(exception_type, format_str, ...)                                                                     \
    do                                                                                                             \
    {                                                                                                              \
        auto formatted_msg = fmt::format("\033[31m[{}:{}:{}] {}\033[0m", origin::basename(__FILE__), __FUNCTION__, \
                                         __LINE__, fmt::format(format_str, ##__VA_ARGS__));                        \
        throw exception_type(formatted_msg);                                                                       \
    } while (0)

// 带条件的THROW宏
#define THROW_IF(condition, exception_type, format_str, ...)  \
    do                                                        \
    {                                                         \
        if (condition)                                        \
        {                                                     \
            THROW(exception_type, format_str, ##__VA_ARGS__); \
        }                                                     \
    } while (0)

// 常用异常类型的简化宏
#define THROW_INVALID_ARG(format_str, ...) THROW(std::invalid_argument, format_str, ##__VA_ARGS__)
#define THROW_RUNTIME_ERROR(format_str, ...) THROW(std::runtime_error, format_str, ##__VA_ARGS__)
#define THROW_LOGIC_ERROR(format_str, ...) THROW(std::logic_error, format_str, ##__VA_ARGS__)
#define THROW_OUT_OF_RANGE(format_str, ...) THROW(std::out_of_range, format_str, ##__VA_ARGS__)
#define THROW_BAD_ALLOC(format_str, ...) THROW(std::bad_alloc, format_str, ##__VA_ARGS__)

#endif  // __ORIGIN_DL_EXCEPTION_H__