#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA幂运算算子实现
 * @details 实现高效的GPU幂运算
 *
 * ============================================================================
 * PyTorch pow行为详解
 * ============================================================================
 *
 * PyTorch的pow算子支持多种幂运算场景：
 *
 * 1. 标量幂运算：
 *    - 张量的每个元素都进行相同的幂运算
 *    - 支持整数和浮点数指数
 *
 *    示例：
 *    ```python
 *    import torch
 *    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
 *    result = torch.pow(x, 2)  # 每个元素平方
 *    result = x ** 2  # 等价写法
 *    ```
 *
 * 2. 张量幂运算：
 *    - 两个张量对应元素进行幂运算
 *    - 支持广播
 *
 *    示例：
 *    ```python
 *    x = torch.tensor([2.0, 3.0, 4.0])
 *    y = torch.tensor([2.0, 1.0, 0.5])
 *    result = torch.pow(x, y)  # [4.0, 3.0, 2.0]
 *    ```
 *
 * 3. 特殊值处理：
 *    - 0的0次方 = 1
 *    - 负数的非整数次方 = NaN
 *    - 0的负数次方 = inf
 *
 * ============================================================================
 * 当前实现说明
 * ============================================================================
 *
 * 当前实现采用基于CUDA内核的幂运算策略：
 * - 实现高效的标量幂运算内核
 * - 支持所有数值类型
 * - 使用优化的数学函数
 * - 与CPU版本保持一致的行为
 *
 * ============================================================================
 * 未来优化计划
 * ============================================================================
 *
 * 计划实现更高级的优化：
 * 1. 实现张量幂运算（两个张量对应元素）
 * 2. 使用更高效的数学库
 * 3. 支持复数幂运算
 * 4. 实现更复杂的广播逻辑
 *
 */

/**
 * @brief 标量幂运算CUDA内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param exponent 指数
 * @param n 元素总数
 */
template <typename T>
__global__ void pow_scalar_kernel(const T *__restrict__ input, T *__restrict__ output, T exponent, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        T base = input[idx];

        // 直接使用CUDA标准幂运算，与CPU版本保持一致
        // 对于float类型使用powf，对于double类型使用pow
        if (sizeof(T) == sizeof(float))
        {
            output[idx] = powf(static_cast<float>(base), static_cast<float>(exponent));
        }
        else
        {
            output[idx] = pow(static_cast<double>(base), static_cast<double>(exponent));
        }
    }
}

/**
 * @brief 张量幂运算CUDA内核
 * @tparam T 数据类型
 * @param base 底数张量
 * @param exponent 指数张量
 * @param output 输出数据
 * @param n 元素总数
 */
template <typename T>
__global__ void pow_tensor_kernel(const T *__restrict__ base,
                                  const T *__restrict__ exponent,
                                  T *__restrict__ output,
                                  size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        T b = base[idx];
        T e = exponent[idx];

        // 直接使用CUDA标准幂运算，与CPU版本保持一致
        // 对于float类型使用powf，对于double类型使用pow
        if (sizeof(T) == sizeof(float))
        {
            output[idx] = powf(static_cast<float>(b), static_cast<float>(e));
        }
        else
        {
            output[idx] = pow(static_cast<double>(b), static_cast<double>(e));
        }
    }
}

/**
 * @brief 启动标量幂运算内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param exponent 指数
 * @param n 元素总数
 */
template <typename T>
void launch_pow_scalar_kernel(const T *input, T *output, T exponent, size_t n)
{
    const int block_size = 256;
    const int grid_size  = (n + block_size - 1) / block_size;

    pow_scalar_kernel<T><<<grid_size, block_size>>>(input, output, exponent, n);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA pow scalar kernel launch failed: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief 启动张量幂运算内核
 * @tparam T 数据类型
 * @param base 底数张量
 * @param exponent 指数张量
 * @param output 输出数据
 * @param n 元素总数
 */
template <typename T>
void launch_pow_tensor_kernel(const T *base, const T *exponent, T *output, size_t n)
{
    const int block_size = 256;
    const int grid_size  = (n + block_size - 1) / block_size;

    pow_tensor_kernel<T><<<grid_size, block_size>>>(base, exponent, output, n);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA pow tensor kernel launch failed: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief CUDA标量幂运算算子实现
 * @param mat 输入矩阵
 * @param exponent 指数
 * @return 幂运算结果矩阵
 */
std::unique_ptr<Mat> pow(const OriginMat &mat, data_t exponent)
{
    // 验证设备类型
    validation::validate_cuda_device(mat, "pow");

    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype(), mat.device());

    // 使用类型分发器执行标量幂运算
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        launch_pow_scalar_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), static_cast<T>(exponent), mat.elements());
    });

    cudaDeviceSynchronize();
    return result;
}

/**
 * @brief CUDA张量幂运算算子实现
 * @param base 底数矩阵
 * @param exponent 指数矩阵
 * @return 幂运算结果矩阵
 */
std::unique_ptr<Mat> pow(const OriginMat &base, const OriginMat &exponent)
{
    // 验证输入
    if (base.dtype() != exponent.dtype())
    {
        THROW_INVALID_ARG("Data type mismatch in CUDA pow: {} vs {}", dtype_to_string(base.dtype()),
                          dtype_to_string(exponent.dtype()));
    }

    // 验证设备类型
    validation::validate_same_cuda_device(base, exponent, "pow");

    // 验证形状兼容性（简化实现，要求相同形状）
    if (base.shape() != exponent.shape())
    {
        THROW_INVALID_ARG("Shape mismatch in CUDA pow: {} vs {}", base.shape().to_string(),
                          exponent.shape().to_string());
    }

    auto result = std::make_unique<OriginMat>(base.shape(), base.dtype(), base.device());

    // 使用类型分发器执行张量幂运算
    device_common::TypeDispatcher::dispatch_void(base.dtype(), [&]<typename T>() {
        launch_pow_tensor_kernel(base.data_ptr<T>(), exponent.data_ptr<T>(), result->data_ptr<T>(), base.elements());
    });

    cudaDeviceSynchronize();
    return result;
}

}  // namespace cuda
}  // namespace origin
