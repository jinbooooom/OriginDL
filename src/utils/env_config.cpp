#include "origin/utils/env_config.h"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace origin
{
namespace utils
{

int32_t EnvConfig::read_kernel_algo_from_env()
{
    constexpr int32_t kDefaultKernelAlgo = 6666;  // 默认值：自动选择

    const char *env_val = std::getenv("ORIGIN_KERNEL_ALGO");
    if (env_val == nullptr || env_val[0] == '\0')
    {
        return kDefaultKernelAlgo;
    }

    // 尝试解析为整数
    char *end_ptr = nullptr;
    long parsed_value = std::strtol(env_val, &end_ptr, 10);

    // 检查是否成功解析（end_ptr 应该指向字符串末尾）
    if (end_ptr != nullptr && *end_ptr == '\0' && parsed_value >= INT32_MIN && parsed_value <= INT32_MAX)
    {
        return static_cast<int32_t>(parsed_value);
    }

    // 解析失败，返回默认值
    return kDefaultKernelAlgo;
}

EnvConfig::EnvConfig() : kernel_algo_(read_kernel_algo_from_env())
{
}

}  // namespace utils
}  // namespace origin
