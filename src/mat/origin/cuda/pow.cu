#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA幂运算算子实现
 * @details 实现高效的GPU幂运算，支持不同底数和指数类型的组合
 *
 * ============================================================================
 * 设计架构说明
 * ============================================================================
 *
 * 1. 核函数层面，用完全特化的核函数设计，提供最佳性能：
 *    - pow_scalar_kernel_float: 专门处理float类型，使用powf函数（高性能）
 *    - pow_scalar_kernel_double: 专门处理double类型，使用pow函数（高精度）
 *    - 使用__restrict__关键字优化内存访问，核函数内无任何类型判断，纯计算，性能最优
 *
 * 2. 类型选择规则：
 *    - 如果底数类型T或指数类型U中有一个是double，使用double版本（pow函数）
 *    - 否则使用float版本（powf函数）
 *    - 这个规则在launch函数中通过if constexpr实现，编译时确定，零运行时开销
 *
 * 3. 支持的类型组合：
 *    - float + float → 使用powf（性能最优）
 *    - float + double → 使用pow（精度优先）
 *    - double + float → 使用pow（精度优先）
 *    - double + double → 使用pow（精度最高）
 *    - int + int → 使用powf（性能优先，精度足够）
 *    - int + float → 使用powf（性能优先）
 *    - int + double → 使用pow（精度优先）
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
 */

/**
 * @brief 标量幂运算CUDA内核 - float版本（使用powf）
 * @details 专门处理float类型的幂运算，使用CUDA的powf函数获得最佳性能
 * @param input 输入数据
 * @param output 输出数据
 * @param exponent 指数
 * @param n 元素总数
 */
__global__ void pow_scalar_kernel_float(const float *__restrict__ input,
                                        float *__restrict__ output,
                                        float exponent,
                                        size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // 使用CUDA的powf函数，性能最优
        output[idx] = powf(input[idx], exponent);
    }
}

/**
 * @brief 标量幂运算CUDA内核 - double版本（使用pow）
 * @details 专门处理double类型的幂运算，使用CUDA的pow函数获得最高精度
 * @param input 输入数据
 * @param output 输出数据
 * @param exponent 指数
 * @param n 元素总数
 */
__global__ void pow_scalar_kernel_double(const double *__restrict__ input,
                                         double *__restrict__ output,
                                         double exponent,
                                         size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // 使用CUDA的pow函数，精度最高
        output[idx] = pow(input[idx], exponent);
    }
}

/**
 * @brief 启动标量幂运算内核
 * @details 根据底数类型T和指数类型U选择合适的核函数
 * 类型选择规则：
 * - 如果T或U中有一个是double类型，使用double版本（pow函数，高精度）
 * - 否则使用float版本（powf函数，高性能）
 * @tparam T 底数数据类型
 * @tparam U 指数数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param exponent 指数
 * @param n 元素总数
 */
template <typename T, typename U>
void launch_pow_scalar_kernel(const T *input, T *output, U exponent, size_t n)
{
    const int block_size = 256;
    const int grid_size  = (n + block_size - 1) / block_size;

    // 类型选择逻辑：只要T或U有一个是double，就使用double版本（pow）
    // 否则使用float版本（powf）
    if constexpr (std::is_same<T, double>::value || std::is_same<U, double>::value)
    {
        // 有double类型，使用double版本（pow函数，高精度）
        pow_scalar_kernel_double<<<grid_size, block_size>>>((const double *)input, (double *)output, (double)exponent, n);
    }
    else
    {
        // 都是float或更小类型，使用float版本（powf函数，高性能）
        pow_scalar_kernel_float<<<grid_size, block_size>>>((const float *)input, (float *)output, (float)exponent, n);
    }
}

/**
 * @brief CUDA标量幂运算算子实现（支持不同类型指数）
 * @details 支持任意数值类型的指数，根据底数和指数类型自动选择最优的核函数
 * 类型选择规则：
 * - 如果底数或指数中有一个是double类型，使用double版本（pow函数，高精度）
 * - 否则使用float版本（powf函数，高性能）
 * @param mat 输入矩阵
 * @param exponent 指数
 * @return 幂运算结果矩阵
 */
auto pow(const OriginMat &base, const Scalar &exponent) -> std::unique_ptr<Mat>
{
    auto result = std::make_unique<OriginMat>(base.shape(), base.dtype(), base.device());

    device_common::TypeDispatcher::dispatch_void(base.dtype(), [&]<typename T>() {
        if (exponent.dtype() == DataType::kFloat64) {
            launch_pow_scalar_kernel(base.data_ptr<T>(), result->data_ptr<T>(),
                                     exponent.to_float64(), base.elements());
        } else {
            launch_pow_scalar_kernel(base.data_ptr<T>(), result->data_ptr<T>(),
                                     exponent.to_float32(), base.elements());
        }
    });

    cudaDeviceSynchronize();
    return result;
}

}  // namespace cuda
}  // namespace origin
