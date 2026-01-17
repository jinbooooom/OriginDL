#ifndef __ORIGIN_DL_MEMORY_POOL_MANAGER_H__
#define __ORIGIN_DL_MEMORY_POOL_MANAGER_H__

#include <memory>
#include <mutex>
#include <vector>
#include "../../basic_types.h"

namespace origin
{

// 前向声明
class MemoryPool;
class Allocator;

/**
 * @brief 内存池管理器（单例）
 *
 * 管理所有设备的内存池：
 * - pools_[0] ~ pools_[N-1]: CUDA 0 ~ CUDA N-1
 * - pools_[N]: CPU
 */
class MemoryPoolManager
{
private:
    std::mutex mutex_;
    std::vector<std::unique_ptr<MemoryPool>> pools_;
    int max_cuda_devices_;

    // 私有构造函数
    MemoryPoolManager();
    ~MemoryPoolManager() = default;

    // 禁止拷贝和赋值
    MemoryPoolManager(const MemoryPoolManager &)            = delete;
    MemoryPoolManager &operator=(const MemoryPoolManager &) = delete;

    /**
     * @brief 获取池的索引
     * @param device_type 设备类型
     * @param device_index 设备索引
     * @return 在 pools_ vector 中的索引
     */
    int get_pool_index(DeviceType device_type, int device_index) const;

    /**
     * @brief 初始化 pools_ vector（检测 CUDA 设备数量）
     */
    void initialize_pools();

public:
    /**
     * @brief 获取单例实例
     */
    static MemoryPoolManager &get_instance();

    /**
     * @brief 获取内存池
     * @param device_type 设备类型
     * @param device_index 设备索引
     * @return 内存池指针
     */
    MemoryPool *get_memory_pool(DeviceType device_type, int device_index);
};

}  // namespace origin

#endif  // __ORIGIN_DL_MEMORY_POOL_MANAGER_H__
