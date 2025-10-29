#include <curand.h>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

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
template <typename T>
__global__ void ones_kernel(T *data, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = static_cast<T>(1);
    }
}

template <typename T>
__global__ void zeros_kernel(T *data, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = static_cast<T>(0);
    }
}

template <typename T>
__global__ void full_kernel(T *data, size_t n, T value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = value;
    }
}

// 显式实例化模板 kernel
template __global__ void ones_kernel<float>(float *data, size_t n);
template __global__ void ones_kernel<double>(double *data, size_t n);
template __global__ void ones_kernel<int32_t>(int32_t *data, size_t n);
template __global__ void ones_kernel<int8_t>(int8_t *data, size_t n);

template __global__ void zeros_kernel<float>(float *data, size_t n);
template __global__ void zeros_kernel<double>(double *data, size_t n);
template __global__ void zeros_kernel<int32_t>(int32_t *data, size_t n);
template __global__ void zeros_kernel<int8_t>(int8_t *data, size_t n);

template __global__ void full_kernel<float>(float *data, size_t n, float value);
template __global__ void full_kernel<double>(double *data, size_t n, double value);
template __global__ void full_kernel<int32_t>(int32_t *data, size_t n, int32_t value);
template __global__ void full_kernel<int8_t>(int8_t *data, size_t n, int8_t value);

/**
 * @brief 启动ones kernel的通用模板函数
 * @tparam T 数据类型
 */
template <typename T>
void launch_ones_kernel(T *data, size_t n)
{
    const size_t block_size = 256;
    const size_t grid_size  = (n + block_size - 1) / block_size;
    ones_kernel<T><<<grid_size, block_size>>>(data, n);
}

/**
 * @brief 启动zeros kernel的通用模板函数
 * @tparam T 数据类型
 */
template <typename T>
void launch_zeros_kernel(T *data, size_t n)
{
    const size_t block_size = 256;
    const size_t grid_size  = (n + block_size - 1) / block_size;
    zeros_kernel<T><<<grid_size, block_size>>>(data, n);
}

/**
 * @brief 启动full kernel的通用模板函数
 * @tparam T 数据类型
 */
template <typename T>
void launch_full_kernel(T *data, size_t n, T value)
{
    const size_t block_size = 256;
    const size_t grid_size  = (n + block_size - 1) / block_size;
    full_kernel<T><<<grid_size, block_size>>>(data, n, value);
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
    size_t n   = shape.elements();

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(),
                                                 [&]<typename T>() { launch_zeros_kernel(static_cast<T *>(data), n); });

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
    size_t n   = shape.elements();

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(),
                                                 [&]<typename T>() { launch_ones_kernel(static_cast<T *>(data), n); });

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

/**
 * @brief 在CUDA设备上创建填充指定值的张量
 */
std::unique_ptr<origin::OriginMat> full(const Shape &shape, const Scalar &scalar, const TensorOptions &options)
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
    size_t n   = shape.elements();

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(
        options.dtype(), [&]<typename T>() { launch_full_kernel(static_cast<T *>(data), n, scalar.to<T>()); });

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

std::unique_ptr<OriginMat> from_memory(const void *data,
                                       DataType user_dtype,
                                       const Shape &shape,
                                       const TensorOptions &options)
{
    // 验证设备
    if (options.device().type() != DeviceType::kCUDA)
    {
        THROW_INVALID_ARG("CUDA from_memory requires CUDA device, got: {}", options.device().to_string());
    }

    // 设置CUDA设备
    set_cuda_device(options.device().index());

    // 创建存储
    size_t size  = shape.elements() * utils::get_dtype_size(options.dtype());
    auto storage = Storage::create(size, options.device().type(), options.device().index());

    // 检查是否需要类型转换
    if (user_dtype == options.dtype())
    {
        // 不需要转换，直接CUDA内存复制
        cudaMemcpy(storage->data(), data, size, cudaMemcpyHostToDevice);
    }
    else
    {
        // 需要类型转换，先在CPU上进行转换，然后复制到CUDA，TODO:未来放到 cuda 中去做，避免了一次拷贝。
        // 为什么需要类型转换：
        // 假设用户传入的数据是float64类型，但是希望创建的是int32类型的张量，那么需要将float64类型转换为int32类型。
        // float64 用 8字节表示，int32 用 4字节表示，
        // 不转换会导致把一个 float64 的高4位和低4位分别当做一个int32的值进行重解释，显然就是错误的。
        size_t temp_size  = shape.elements() * utils::get_dtype_size(options.dtype());
        void *temp_buffer = malloc(temp_size);

        // 使用TypeDispatcher进行转换
        device_common::TypeDispatcher::dispatch_void(user_dtype, [&]<typename T>() {
            const T *user_data = static_cast<const T *>(data);
            // 使用TypeDispatcher进行目标类型转换
            device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename U>() {
                U *temp_data = static_cast<U *>(temp_buffer);
                for (size_t i = 0; i < shape.elements(); ++i)
                {
                    temp_data[i] = static_cast<U>(user_data[i]);
                }
            });
        });

        // 复制到CUDA设备
        cudaMemcpy(storage->data(), temp_buffer, temp_size, cudaMemcpyHostToDevice);
        free(temp_buffer);
    }

    return std::make_unique<OriginMat>(storage, shape, options.dtype());
}

}  // namespace cuda
}  // namespace origin
