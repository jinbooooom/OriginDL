#ifndef __ORIGIN_DL_CUDA_H__
#define __ORIGIN_DL_CUDA_H__

// CUDA 公共 API 头文件
// 只暴露用户需要的函数，内部实现细节保留在 cuda_utils.cuh 中

#include <iostream>
#include "../utils/exception.h"

#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#    include "../mat/origin/cuda/cuda_utils.cuh"
#endif

namespace origin
{
namespace cuda
{

/**
 * @brief 检查CUDA设备是否可用
 * @return 如果CUDA可用返回true，否则返回false
 * @details 这是 torch.cuda.is_available() 的对应API
 */
inline bool is_available()
{
#ifdef WITH_CUDA
    return is_cuda_available();
#else
    return false;
#endif
}

/**
 * @brief 获取可用的CUDA设备数量
 * @return 可用的CUDA设备数量
 * @details 这是 torch.cuda.device_count() 的对应API
 */
inline int device_count()
{
#ifdef WITH_CUDA
    return get_cuda_device_count();
#else
    return 0;
#endif
}

/**
 * @brief 获取当前选定的CUDA设备索引
 * @return 当前CUDA设备索引
 * @details 这是 torch.cuda.current_device() 的对应API，如果未启用CUDA支持，则抛出异常
 */
inline int current_device()
{
#ifdef WITH_CUDA
    return get_current_cuda_device();
#else
    THROW_RUNTIME_ERROR("CUDA support is not enabled. Please rebuild with --cuda flag: ./build.sh origin --cuda");
    return -1;
#endif
}

/**
 * @brief 设置当前CUDA设备
 * @param device_id 要设置的设备ID
 * @details 这是 torch.cuda.set_device(device) 的对应API
 * @throws std::runtime_error 如果编译时未启用CUDA支持，则抛出异常
 */
inline void set_device(int device_id)
{
#ifdef WITH_CUDA
    set_cuda_device(device_id);
#else
    (void)device_id;
    THROW_RUNTIME_ERROR("CUDA support is not enabled. Please rebuild with --cuda flag: ./build.sh origin --cuda");
#endif
}

/**
 * @brief 打印CUDA设备信息
 * @details 打印所有可用CUDA设备的信息，包括设备名称、计算能力、内存等，如果未启用CUDA支持，则打印提示信息
 */
inline void device_info()
{
#ifdef WITH_CUDA
    print_cuda_device_info();
#else
    std::cout << "CUDA support is not enabled. Please rebuild with --cuda flag: ./build.sh origin --cuda" << std::endl;
#endif
}

/**
 * @brief 同步CUDA设备，等待所有CUDA操作完成
 * @details 这是 cudaDeviceSynchronize() 的封装，用于确保所有CUDA操作完成
 *          如果未启用CUDA支持，则抛异常
 */
inline void synchronize()
{
#ifdef WITH_CUDA
    cudaDeviceSynchronize();
#else
    THROW_RUNTIME_ERROR("CUDA support is not enabled. Please rebuild with --cuda flag: ./build.sh origin --cuda");
    return;
#endif
}

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_H__
