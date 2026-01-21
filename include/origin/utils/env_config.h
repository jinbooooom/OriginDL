#ifndef __ORIGIN_DL_ENV_CONFIG_H__
#define __ORIGIN_DL_ENV_CONFIG_H__

#include <cstdint>

namespace origin
{
namespace utils
{

/**
 * @brief 环境变量配置管理单例类
 * @details 用于读取和缓存环境变量配置，首次构建时读取环境变量并缓存，
 *          后续访问不会重新解析环境变量。
 *
 * 当前支持的环境变量：
 * - ORIGIN_KERNEL_ALGO: 内核算法版本号（数字），默认值为 6666
 *
 * @example
 *   export ORIGIN_KERNEL_ALGO=0    # 使用算法版本 0
 *   export ORIGIN_KERNEL_ALGO=6666  # 使用自动选择（默认）
 */
class EnvConfig
{
private:
    /**
     * @brief 从环境变量 ORIGIN_KERNEL_ALGO 读取内核算法版本号
     * @details 如果环境变量未设置或无效，返回默认值 6666
     * @return 算法版本号
     */
    static int32_t read_kernel_algo_from_env();

    // 禁止拷贝和赋值
    EnvConfig(const EnvConfig &)            = delete;
    EnvConfig &operator=(const EnvConfig &) = delete;

    // 私有构造函数，确保单例
    EnvConfig();

    // 缓存的算法版本号
    int32_t kernel_algo_;

public:
    /**
     * @brief 获取单例实例
     * @return EnvConfig 单例实例的引用
     */
    static EnvConfig &GetInstance()
    {
        static EnvConfig instance;
        return instance;
    }

    /**
     * @brief 获取内核算法版本号
     * @return 算法版本号（从环境变量读取，默认 6666）
     */
    int32_t kernel_algo() const { return kernel_algo_; }
};

}  // namespace utils
}  // namespace origin

#endif  // __ORIGIN_DL_ENV_CONFIG_H__
