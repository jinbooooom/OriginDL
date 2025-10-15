#include "origin/mat/origin/allocator.h"
#include <cstdlib>  // For std::aligned_alloc, std::free
#include <stdexcept>
#include "origin/utils/exception.h"

// For CUDA
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

// CUDAAllocator implementations
void *CUDAAllocator::allocate(size_t size)
{
#ifdef WITH_CUDA
    void *ptr       = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA memory allocation failed: {} (requested size: {} bytes)", cudaGetErrorString(err),
                            size);
    }
    return ptr;
#else
    THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
}

void CUDAAllocator::deallocate(void *ptr)
{
#ifdef WITH_CUDA
    if (ptr != nullptr)
    {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess)
        {
            THROW_RUNTIME_ERROR("CUDA memory deallocation failed: {}", cudaGetErrorString(err));
        }
    }
#else
    THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
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
#ifdef WITH_CUDA
        return std::make_unique<CUDAAllocator>(device_index);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_INVALID_ARG("Unsupported device type for allocator: {}", static_cast<int>(device_type));
    }
}

}  // namespace origin