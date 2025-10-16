#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"
#include <random>
#include <curand.h>
#include <vector>
#include <type_traits>
#include <memory>
#include <utility>

namespace origin
{
namespace cuda
{
    // 引入需要的类型
    using origin::OriginMat;
    using std::unique_ptr;

/**
 * @brief CUDA kernel 设置所有元素为1 (模板版本)
 * @tparam T 数据类型
 */
template<typename T>
__global__ void ones_kernel(T* data, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = static_cast<T>(1);
    }
}

template<typename T>
__global__ void zeros_kernel(T* data, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = static_cast<T>(0);
    }
}

template<typename T>
__global__ void full_kernel(T* data, size_t n, T value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = value;
    }
}


// 显式实例化模板 kernel
template __global__ void ones_kernel<float>(float* data, size_t n);
template __global__ void ones_kernel<double>(double* data, size_t n);
template __global__ void ones_kernel<int32_t>(int32_t* data, size_t n);
template __global__ void ones_kernel<int8_t>(int8_t* data, size_t n);

template __global__ void zeros_kernel<float>(float* data, size_t n);
template __global__ void zeros_kernel<double>(double* data, size_t n);
template __global__ void zeros_kernel<int32_t>(int32_t* data, size_t n);
template __global__ void zeros_kernel<int8_t>(int8_t* data, size_t n);

template __global__ void full_kernel<float>(float* data, size_t n, float value);
template __global__ void full_kernel<double>(double* data, size_t n, double value);
template __global__ void full_kernel<int32_t>(int32_t* data, size_t n, int32_t value);
template __global__ void full_kernel<int8_t>(int8_t* data, size_t n, int8_t value);

/**
 * @brief 启动ones kernel的通用模板函数
 * @tparam T 数据类型
 */
template<typename T>
void launch_ones_kernel(T* data, size_t n)
{
    const size_t block_size = 256;
    const size_t grid_size = (n + block_size - 1) / block_size;
    ones_kernel<T><<<grid_size, block_size>>>(data, n);
    
    // 检查kernel启动是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to launch ones kernel: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief 启动zeros kernel的通用模板函数
 * @tparam T 数据类型
 */
template<typename T>
void launch_zeros_kernel(T* data, size_t n)
{
    const size_t block_size = 256;
    const size_t grid_size = (n + block_size - 1) / block_size;
    zeros_kernel<T><<<grid_size, block_size>>>(data, n);
    
    // 检查kernel启动是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to launch zeros kernel: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief 启动full kernel的通用模板函数
 * @tparam T 数据类型
 */
template<typename T>
void launch_full_kernel(T* data, size_t n, T value)
{
    const size_t block_size = 256;
    const size_t grid_size = (n + block_size - 1) / block_size;
    full_kernel<T><<<grid_size, block_size>>>(data, n, value);
    
    // 检查kernel启动是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to launch full kernel: {}", cudaGetErrorString(err));
    }
}



/**
 * @brief 在CUDA设备上创建零张量
 */
std::unique_ptr<origin::OriginMat> zeros(const Shape &shape, const TensorOptions &options)
{
    // 验证设备
    if (options.device().type() != DeviceType::kCUDA)
    {
        THROW_INVALID_ARG("CUDA zeros requires CUDA device, got: {}", options.device().to_string());
    }
    
    // 设置CUDA设备
    set_cuda_device(options.device().index());
    
    unique_ptr<OriginMat> result(new OriginMat(shape, options.dtype(), options.device()));
    
    // 获取数据指针
    void *data = result->storage()->data();
    size_t n = shape.elements();
    
    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        launch_zeros_kernel(static_cast<T*>(data), n);
    });
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    return result;
}

/**
 * @brief 在CUDA设备上创建全1张量
 */
std::unique_ptr<origin::OriginMat> ones(const Shape &shape, const TensorOptions &options)
{
    // 验证设备
    if (options.device().type() != DeviceType::kCUDA)
    {
        THROW_INVALID_ARG("CUDA ones requires CUDA device, got: {}", options.device().to_string());
    }
    
    // 设置CUDA设备
    set_cuda_device(options.device().index());
    
    unique_ptr<OriginMat> result(new OriginMat(shape, options.dtype(), options.device()));
    
    // 获取数据指针
    void *data = result->storage()->data();
    size_t n = shape.elements();
    
    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        launch_ones_kernel(static_cast<T*>(data), n);
    });
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    return result;
}

/**
 * @brief 在CUDA设备上创建填充指定值的张量
 */
std::unique_ptr<origin::OriginMat> full(const Shape &shape, double value, const TensorOptions &options)
{
    // 验证设备
    if (options.device().type() != DeviceType::kCUDA)
    {
        THROW_INVALID_ARG("CUDA full requires CUDA device, got: {}", options.device().to_string());
    }
    
    // 设置CUDA设备
    set_cuda_device(options.device().index());
    
    unique_ptr<OriginMat> result(new OriginMat(shape, options.dtype(), options.device()));
    
    // 获取数据指针
    void *data = result->storage()->data();
    size_t n = shape.elements();
    
    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        launch_full_kernel(static_cast<T*>(data), n, static_cast<T>(value));
    });
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    return result;
}

}  // namespace cuda
}  // namespace origin
