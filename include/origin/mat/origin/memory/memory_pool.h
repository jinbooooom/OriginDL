#ifndef __ORIGIN_DL_MEMORY_POOL_H__
#define __ORIGIN_DL_MEMORY_POOL_H__

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "../../basic_types.h"

namespace origin
{

// 前向声明
class Allocator;

/**
 * @brief 通用内存池类
 *
 * 提供统一的内存池逻辑，所有设备（CPU/CUDA）共享相同的实现
 * 差异仅在于底层 Allocator 的具体分配/释放函数
 */
class MemoryPool
{
private:
    /**
     * @brief 内存块结构
     */
    struct MemoryBlock
    {
        void *ptr;    // 内存指针
        size_t size;  // 内存块大小
        bool in_use;  // 是否正在使用

        MemoryBlock(void *p, size_t s) : ptr(p), size(s), in_use(false) {}
    };

    // 底层分配器（CPUAllocator 或 CUDAAllocator）
    std::unique_ptr<Allocator> backend_;

    // 内存池：块大小 -> 内存块列表
    std::multimap<size_t, MemoryBlock> pool_;

    // 指针到块的映射（用于快速查找）
    std::unordered_map<void *, std::multimap<size_t, MemoryBlock>::iterator> ptr_to_block_;

    // 当前缓存大小
    size_t current_cache_size_;

    // 最大缓存大小
    size_t max_cache_size_;

    // 线程安全锁
    mutable std::mutex mutex_;

    // 设备类型和索引（用于标识）
    DeviceType device_type_;
    int device_index_;

    /**
     * @brief 计算分桶大小
     */
    size_t get_bin_size(size_t size) const;

    /**
     * @brief 清理超出容量上限的内存块
     */
    void trim_cache();

public:
    /**
     * @brief 构造函数
     * @param device_type 设备类型
     * @param device_index 设备索引
     * @param max_cache_size 最大缓存大小（默认 512MB）
     *
     * @note MemoryPool 内部根据 device_type 和 device_index 创建 Allocator
     */
    MemoryPool(DeviceType device_type, int device_index, size_t max_cache_size = 512 * 1024 * 1024);

    /**
     * @brief 析构函数：释放所有缓存的内存块
     */
    ~MemoryPool();

    /**
     * @brief 分配内存
     * @param size 要分配的字节数
     * @return 分配的内存指针
     */
    void *malloc(size_t size);

    /**
     * @brief 释放内存
     * @param ptr 要释放的内存指针
     */
    void free(void *ptr);

    /**
     * @brief 获取设备类型
     */
    DeviceType device_type() const { return device_type_; }

    /**
     * @brief 获取设备索引
     */
    int device_index() const { return device_index_; }
};

}  // namespace origin

#endif  // __ORIGIN_DL_MEMORY_POOL_H__
