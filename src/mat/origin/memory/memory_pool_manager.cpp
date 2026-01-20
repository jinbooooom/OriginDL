#include "origin/mat/origin/memory/memory_pool_manager.h"
#include "origin/mat/origin/memory/allocator.h"
#include "origin/mat/origin/memory/memory_pool.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#endif

namespace origin
{

MemoryPoolManager::MemoryPoolManager() : max_cuda_devices_(0)
{
    // 初始化时创建 vector
    initialize_pools();
}

MemoryPoolManager &MemoryPoolManager::get_instance()
{
    static MemoryPoolManager instance;
    return instance;
}

void MemoryPoolManager::initialize_pools()
{
    if (!pools_.empty())
    {
        return;  // 已经初始化
    }

#ifdef WITH_CUDA
    // 检测 CUDA 设备数量
    int device_count = 0;
    cudaError_t err  = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        // CUDA 不可用，设置为 0
        device_count = 0;
    }
    max_cuda_devices_ = device_count;
#else
    max_cuda_devices_ = 0;
#endif

    // 初始化 vector：CUDA 设备数 + 1（CPU）
    int pool_count = max_cuda_devices_ + 1;
    pools_.clear();
    pools_.reserve(pool_count);
    pools_.resize(pool_count);
}

int MemoryPoolManager::get_pool_index(DeviceType device_type, int device_index) const
{
    if (device_type == DeviceType::kCPU)
    {
        // CPU 在最后一个索引
        return static_cast<int>(pools_.size()) - 1;
    }
    else if (device_type == DeviceType::kCUDA)
    {
        // CUDA 设备索引直接对应
        if (unlikely(device_index < 0 || device_index >= max_cuda_devices_))
        {
            THROW_INVALID_ARG("Invalid CUDA device index: {}, max devices: {}", device_index, max_cuda_devices_);
        }
        return device_index;
    }
    else
    {
        THROW_INVALID_ARG("Unsupported device type: {}", static_cast<int>(device_type));
    }
}

MemoryPool *MemoryPoolManager::get_memory_pool(DeviceType device_type, int device_index)
{
    std::lock_guard<std::mutex> lock(mutex_);

    int index = get_pool_index(device_type, device_index);
    if (!pools_[index])
    {
        pools_[index] = std::make_unique<MemoryPool>(device_type, device_index);
    }

    return pools_[index].get();
}

}  // namespace origin
