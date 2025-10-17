#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include <iostream>
#include "origin/utils/exception.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/branch_prediction.h"

namespace origin
{
namespace cuda
{

dim3 get_optimal_block_size(size_t n)
{
    // 根据数据大小选择最优的线程块大小
    if (n < 256)
    {
        return dim3(n);
    }
    else if (n < 1024)
    {
        return dim3(256);
    }
    else
    {
        return dim3(512);
    }
}

dim3 get_optimal_grid_size(size_t n, dim3 block_size)
{
    return dim3((n + block_size.x - 1) / block_size.x);
}

bool is_cuda_available()
{
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

void print_cuda_device_info()
{
    if (unlikely(!is_cuda_available()))
    {
        std::cout << "CUDA is not available" << std::endl;
        return;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA devices available: " << device_count << std::endl;

    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}

int get_cuda_device_count()
{
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (unlikely(err != cudaSuccess))
    {
        return 0;
    }
    return device_count;
}

void set_cuda_device(int device_id)
{
    cudaError_t err = cudaSetDevice(device_id);
    if (unlikely(err != cudaSuccess))
    {
        THROW_RUNTIME_ERROR("Failed to set CUDA device {}: {}", device_id, cudaGetErrorString(err));
    }
}

int get_current_cuda_device()
{
    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (unlikely(err != cudaSuccess))
    {
        THROW_RUNTIME_ERROR("Failed to get current CUDA device: {}", cudaGetErrorString(err));
    }
    return device_id;
}

size_t get_type_size(DataType dtype)
{
    return device_common::TypeDispatcher::dispatch(dtype, [&]<typename T>() -> size_t {
        return sizeof(T);
    });
}

}  // namespace cuda
}  // namespace origin
