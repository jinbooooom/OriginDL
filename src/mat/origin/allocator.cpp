#include "origin/mat/origin/allocator.h"
#include <cstdlib>  // For std::aligned_alloc, std::free
#include <stdexcept>

// For CUDA (empty implementations for now)
#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#endif

namespace origin
{

// CPUAllocator implementations
void *CPUAllocator::allocate(size_t size)
{
    // Allocate 64-byte aligned memory for CPU
    void *ptr = std::aligned_alloc(64, size);
    if (ptr == nullptr)
    {
        throw std::bad_alloc();
    }
    return ptr;
}

void CPUAllocator::deallocate(void *ptr)
{
    std::free(ptr);
}

// CUDAAllocator implementations (empty for now)
void *CUDAAllocator::allocate(size_t size)
{
    // DL_CRITICAL_THROW("CUDA is not supported yet."); // Temporarily removed
    throw std::runtime_error("CUDA is not supported yet.");
    return nullptr;
}

void CUDAAllocator::deallocate(void *ptr)
{
    // DL_CRITICAL_THROW("CUDA is not supported yet."); // Temporarily removed
    throw std::runtime_error("CUDA is not supported yet.");
}

// AllocatorFactory implementation
std::unique_ptr<Allocator> AllocatorFactory::create_allocator(DeviceType device_type, int device_index)
{
    if (device_type == DeviceType::kCPU)
    {
        return std::make_unique<CPUAllocator>();
    }
    else if (device_type == DeviceType::kCUDA)
    {
        return std::make_unique<CUDAAllocator>(device_index);
    }
    else
    {
        throw std::invalid_argument("Unsupported device type for allocator.");
    }
}

}  // namespace origin