#include "origin/mat/origin/memory/allocator.h"
#include <cstdlib>  // For std::aligned_alloc, std::free
#include <stdexcept>
#include "origin/utils/exception.h"

// For CUDA
#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#endif

namespace origin
{

// 内存对齐常量
// 为什么选择128字节对齐？
// 1. CUDA合并内存访问（Coalesced Memory Access）：
//    CUDA的全局内存访问以128字节为基本单位（cache line大小）
//    一个warp（32个线程）访问连续内存时，如果地址对齐到128字节边界，可以合并为一次128字节的传输
//    不对齐会导致多次内存事务，严重影响性能
// 2. CUDA向量化访问：
//    代码中使用了float4（16字节）、double2（16字节）等向量类型
//    128字节对齐可以确保所有向量类型都能正确对齐访问
// 3. CPU兼容性：
//    虽然CPU的cache line通常是64字节，但128字节对齐仍然有效（是64的倍数）
//    128字节对齐对CPU的SIMD操作（如AVX-512）也有好处
// 4. 内存效率平衡：
//    256字节对齐会造成更多内存浪费（例如100字节请求会浪费156字节）
//    128字节对齐在性能和内存效率之间取得良好平衡
//    cudaMalloc返回的指针虽然至少256字节对齐，但size参数对齐到128字节已足够
#ifndef MEMORY_ALIGNMENT
#    define MEMORY_ALIGNMENT 128
#endif

// CPUAllocator implementations
void *CPUAllocator::allocate(size_t size)
{
    // 为CPU分配对齐的内存
    void *ptr = std::aligned_alloc(MEMORY_ALIGNMENT, size);
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
    cudaError_t err = cudaSetDevice(device_index_);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA set device failed: {} (device index: {})", cudaGetErrorString(err), device_index_);
    }
    // CUDA内存对齐：确保size对齐以获得最优性能
    // MEMORY_ALIGNMENT字节对齐足够满足：
    // 合并内存访问（128字节cache line）
    // 向量化访问（float4 = 16字节，可以放入对齐边界内）
    const size_t alignment = MEMORY_ALIGNMENT;
    size_t aligned_size    = (size + alignment - 1) & ~(alignment - 1);
    void *ptr              = nullptr;
    err                    = cudaMalloc(&ptr, aligned_size);
    if (err != cudaSuccess)
    {
        cudaDeviceProp prop = {};
        cudaGetDeviceProperties(&prop, device_index_);
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        THROW_RUNTIME_ERROR(
            "CUDA memory allocation failed: {} (requested size: {} bytes, aligned size: {} bytes). Device[{}] {}: "
            "totalGlobalMem={} bytes, cudaMemGetInfo free={} bytes total={} bytes",
            cudaGetErrorString(err), size, aligned_size, device_index_, prop.name, prop.totalGlobalMem, free_mem,
            total_mem);
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
        cudaError_t err = cudaSetDevice(device_index_);
        if (err != cudaSuccess)
        {
            THROW_RUNTIME_ERROR("CUDA set device failed: {} (device index: {})", cudaGetErrorString(err),
                                device_index_);
        }
        err = cudaFree(ptr);
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