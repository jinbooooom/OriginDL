#include "origin/mat/origin/memory/memory_pool.h"
#include "origin/mat/origin/memory/allocator.h"
#include "origin/utils/exception.h"
#include "origin/utils/log.h"

namespace origin
{

MemoryPool::MemoryPool(DeviceType device_type, int device_index, size_t max_cache_size)
    : max_cache_size_(max_cache_size), current_cache_size_(0), device_type_(device_type), device_index_(device_index)
{
    // 根据设备类型和索引创建底层分配器
    backend_ = AllocatorFactory::create_allocator(device_type, device_index);
    if (!backend_)
    {
        THROW_INVALID_ARG("MemoryPool: failed to create backend allocator for device type {} index {}",
                          static_cast<int>(device_type), device_index);
    }
}

MemoryPool::~MemoryPool()
{
    std::lock_guard<std::mutex> lock(mutex_);
    // 释放所有缓存的内存块
    for (auto &pair : pool_)
    {
        if (pair.second.ptr != nullptr)
        {
            backend_->deallocate(pair.second.ptr);
        }
    }
    pool_.clear();
    ptr_to_block_.clear();
}

size_t MemoryPool::get_bin_size(size_t size) const
{
    // 混合策略：小内存按2的幂分桶，大内存按范围分桶
    const size_t small_memory_threshold = 512 * 1024;  // 512KB

    if (size < small_memory_threshold)
    {
        // 小内存：按2的幂分桶，最小256B
        size_t bin_size = 256;
        while (bin_size < size)
        {
            bin_size *= 2;
        }
        return bin_size;
    }
    else
    {
        // 大内存：按范围分桶
        if (size < 1024 * 1024)
        {
            return 1024 * 1024;  // 512KB-1MB -> 1MB桶
        }
        else if (size < 4 * 1024 * 1024)
        {
            return 4 * 1024 * 1024;  // 1MB-4MB -> 4MB桶
        }
        else if (size < 16 * 1024 * 1024)
        {
            return 16 * 1024 * 1024;  // 4MB-16MB -> 16MB桶
        }
        else
        {
            // 大于16MB：按16MB对齐
            return ((size + 16 * 1024 * 1024 - 1) / (16 * 1024 * 1024)) * 16 * 1024 * 1024;
        }
    }
}

void MemoryPool::trim_cache()
{
    // 如果超过上限，释放空闲块
    while (current_cache_size_ > max_cache_size_ && !pool_.empty())
    {
        bool freed = false;
        for (auto it = pool_.begin(); it != pool_.end();)
        {
            if (!it->second.in_use)
            {
                size_t block_size = it->second.size;
                void *ptr         = it->second.ptr;

                ptr_to_block_.erase(ptr);
                backend_->deallocate(ptr);
                current_cache_size_ -= block_size;
                it    = pool_.erase(it);
                freed = true;
                break;
            }
            else
            {
                ++it;
            }
        }
        if (!freed)
        {
            break;  // 没有空闲块了
        }
    }
}

void *MemoryPool::malloc(size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);

    size_t bin_size = get_bin_size(size);

    // 在池中查找空闲块
    auto it = pool_.lower_bound(bin_size);
    while (it != pool_.end())
    {
        if (!it->second.in_use && it->second.size >= size)
        {
            // 找到合适的块，复用
            it->second.in_use = true;
            return it->second.ptr;
        }
        ++it;
    }

    // 缓存未命中，从底层分配器分配
    void *ptr = backend_->allocate(bin_size);
    if (ptr == nullptr)
    {
        THROW_RUNTIME_ERROR("MemoryPool: backend allocation failed for size {}", size);
    }

    // 加入池
    auto insert_it           = pool_.insert({bin_size, MemoryBlock(ptr, bin_size)});
    insert_it->second.in_use = true;
    ptr_to_block_[ptr]       = insert_it;

    current_cache_size_ += bin_size;
    trim_cache();

    return ptr;
}

void MemoryPool::free(void *ptr)
{
    if (ptr == nullptr)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 在映射中查找
    auto ptr_it = ptr_to_block_.find(ptr);
    if (ptr_it != ptr_to_block_.end())
    {
        // 找到块，标记为空闲
        auto block_it = ptr_it->second;
        if (block_it->second.in_use)
        {
            block_it->second.in_use = false;
            // 内存保留在池中，不释放
        }
        else
        {
            // 块已经空闲，不应该发生
            THROW_RUNTIME_ERROR("MemoryPool: free called on unused block");
        }
        return;
    }

    // 未找到（不在池中），直接释放
    // 这种情况不应该发生，但为了安全，直接释放
    logw("MemoryPool: free called on pointer not in pool (ptr={}), directly deallocating", ptr);
    backend_->deallocate(ptr);
}

}  // namespace origin
