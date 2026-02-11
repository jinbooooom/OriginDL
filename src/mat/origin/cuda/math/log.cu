#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief 自然对数运算CUDA内核 - float版本（使用logf）
 * @details 专门处理float类型的对数运算，使用CUDA的logf函数获得最佳性能
 * @param input 输入数据
 * @param output 输出数据
 * @param n 元素总数
 */
__global__ void log_kernel_float(const float *__restrict__ input, float *__restrict__ output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // 使用CUDA的logf函数，性能最优
        output[idx] = logf(input[idx]);
    }
}

/**
 * @brief 自然对数运算CUDA内核 - double版本（使用log）
 * @details 专门处理double类型的对数运算，使用CUDA的log函数获得最高精度
 * @param input 输入数据
 * @param output 输出数据
 * @param n 元素总数
 */
__global__ void log_kernel_double(const double *__restrict__ input, double *__restrict__ output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // 使用CUDA的log函数，精度最高
        output[idx] = log(input[idx]);
    }
}

/**
 * @brief 启动自然对数运算内核
 * @details 根据数据类型T选择合适的核函数
 * 类型选择规则：
 * - 如果T是double类型，使用double版本（log函数，高精度）
 * - 否则使用float版本（logf函数，高性能）
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param n 元素总数
 */
template <typename T>
void launch_log_kernel(const T *input, T *output, size_t n)
{
    const int block_size = 256;
    const int grid_size  = (n + block_size - 1) / block_size;

    // 类型选择逻辑：如果是double，使用double版本（log）
    // 否则使用float版本（logf）
    if constexpr (std::is_same<T, double>::value)
    {
        // double类型，使用double版本（log函数，高精度）
        log_kernel_double<<<grid_size, block_size>>>((const double *)input, (double *)output, n);
    }
    else
    {
        // float类型，使用float版本（logf函数，高性能）
        log_kernel_float<<<grid_size, block_size>>>((const float *)input, (float *)output, n);
    }
}

/**
 * @brief CUDA自然对数运算统一实现（以 e 为底）
 * @details 实现高效的GPU对数运算，支持float32和float64类型
 *
 * ============================================================================
 * 设计架构说明
 * ============================================================================
 *
 * 1. 核函数层面，用完全特化的核函数设计，提供最佳性能：
 *    - log_kernel_float: 专门处理float类型，使用logf函数（高性能）
 *    - log_kernel_double: 专门处理double类型，使用log函数（高精度）
 *    - 使用__restrict__关键字优化内存访问，核函数内无任何类型判断，纯计算，性能最优
 *
 * 2. 类型选择规则：
 *    - 如果数据类型是double，使用double版本（log函数）
 *    - 否则使用float版本（logf函数）
 *    - 这个规则在launch函数中通过if constexpr实现，编译时确定，零运行时开销
 *
 * 3. 支持的类型：
 *    - float32 → 使用logf（性能最优）
 *    - float64 → 使用log（精度最高）
 *
 * ============================================================================
 * PyTorch log行为详解
 * ============================================================================
 *
 * PyTorch的log算子只支持浮点类型，不支持整型：
 * - 输入为整型Tensor时会报错
 * - 支持float32和float64类型
 *
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> log(const OriginMat &mat, OriginMat *out)
{
    VALIDATE_CUDA_DEVICE(mat);
    VALIDATE_FLOAT_DTYPE(mat);

    OriginMat *result_ptr = nullptr;
    std::unique_ptr<OriginMat> result_unique;

    if (out != nullptr)
    {
        if (unlikely(out->shape() != mat.shape() || out->dtype() != mat.dtype() || out->device() != mat.device()))
        {
            THROW_INVALID_ARG(
                "Output tensor mismatch. Expected shape={}, dtype={}, device={}, but got shape={}, "
                "dtype={}, device={}",
                mat.shape().to_string(), dtype_to_string(mat.dtype()), mat.device().to_string(),
                out->shape().to_string(), dtype_to_string(out->dtype()), out->device().to_string());
        }
        result_ptr = out;
    }
    else
    {
        result_unique = std::make_unique<OriginMat>(mat.shape(), mat.dtype(), mat.device());
        result_ptr    = result_unique.get();
    }

    const void *a_data = mat.storage()->data();
    void *c_data       = result_ptr->storage()->data();

    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        launch_log_kernel(static_cast<const T *>(a_data), static_cast<T *>(c_data), mat.elements());
    });

    return result_unique;
}

}  // namespace cuda
}  // namespace origin
