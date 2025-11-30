#ifndef __ORIGIN_DL_CUDA_H__
#define __ORIGIN_DL_CUDA_H__

// CUDA 公共 API 头文件
// 只暴露用户需要的函数，内部实现细节保留在 cuda_utils.cuh 中

#include "../mat/origin/cuda/cuda_utils.cuh"

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
    return is_cuda_available();
}

/**
 * @brief 获取可用的CUDA设备数量
 * @return 可用的CUDA设备数量
 * @details 这是 torch.cuda.device_count() 的对应API
 */
inline int device_count()
{
    return get_cuda_device_count();
}

/**
 * @brief 获取当前选定的CUDA设备索引
 * @return 当前CUDA设备索引
 * @details 这是 torch.cuda.current_device() 的对应API
 */
inline int current_device()
{
    return get_current_cuda_device();
}

/**
 * @brief 设置当前CUDA设备
 * @param device_id 要设置的设备ID
 * @details 这是 torch.cuda.set_device(device) 的对应API
 */
inline void set_device(int device_id)
{
    set_cuda_device(device_id);
}

/**
 * @brief 打印CUDA设备信息
 * @details 打印所有可用CUDA设备的信息，包括设备名称、计算能力、内存等
 */
inline void device_info()
{
    print_cuda_device_info();
}

// 以下函数从 cuda_utils.cuh 中暴露，作为公共API
// 内部实现细节（如 get_optimal_block_size 等）不在此处暴露

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_H__

