#ifndef __ORIGIN_DL_CUDA_UTILS_H__
#define __ORIGIN_DL_CUDA_UTILS_H__

#include <cuda_runtime.h>
#include "../../../utils/exception.h"
#include "../../basic_types.h"

namespace origin
{
namespace cuda
{

// CUDA错误检查宏（用于同步调用）
#define CUDA_CHECK(call)                                                                                 \
    do                                                                                                   \
    {                                                                                                    \
        cudaError_t err = call;                                                                          \
        if (err != cudaSuccess)                                                                          \
        {                                                                                                \
            THROW_RUNTIME_ERROR("CUDA error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                                                                \
    } while (0)

// CUDA异步错误检查宏（用于异步kernel启动后的错误检查，不阻塞）
#define CUDA_CHECK_ASYNC()                                                                                \
    do                                                                                                   \
    {                                                                                                    \
        cudaError_t err = cudaGetLastError();                                                            \
        if (err != cudaSuccess)                                                                          \
        {                                                                                                \
            THROW_RUNTIME_ERROR("CUDA operation failed in {}: {}", __func__, cudaGetErrorString(err));   \
        }                                                                                                \
    } while (0)

// 获取最优的线程块大小
dim3 get_optimal_block_size(size_t n);

// 获取最优的网格大小
dim3 get_optimal_grid_size(size_t n, dim3 block_size);

// 检查CUDA设备是否可用
bool is_cuda_available();

// 获取当前CUDA设备信息
void print_cuda_device_info();

// 获取CUDA设备数量
int get_cuda_device_count();

// 设置CUDA设备
void set_cuda_device(int device_id);

// 获取当前CUDA设备ID
int get_current_cuda_device();

// 获取数据类型大小
size_t get_type_size(DataType dtype);

}  // namespace cuda
}  // namespace origin

#endif
