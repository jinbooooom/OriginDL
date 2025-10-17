#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/utils/exception.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/branch_prediction.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA指数算子实现
 * @details 计算输入张量每个元素的指数值
 */

/**
 * @brief 类型分发器 - 编译时特化
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param c 输出矩阵C
 * @param n 元素总数
 * @param stream CUDA流
 */
template <typename T>
void launch_exp_kernel(const T *a, T *c, size_t n, cudaStream_t stream = 0)
{
    // 根据数据大小选择最优的内核
    if (n < 1024)
    {
        // 小数据量：使用简单内核
        dim3 block = get_optimal_block_size(n);
        dim3 grid  = get_optimal_grid_size(n, block);
        unary_kernel<T, ExpOp><<<grid, block, 0, stream>>>(a, c, n, ExpOp{});
    }
    else if (n < 1024 * 1024)
    {
        // 中等数据量：使用向量化内核
        dim3 block = get_optimal_block_size(n / 4);
        dim3 grid  = get_optimal_grid_size(n / 4, block);
        vectorized_unary_kernel<T, ExpOp><<<grid, block, 0, stream>>>(a, c, n, ExpOp{});
    }
    else
    {
        dim3 block = get_optimal_block_size(n);
        dim3 grid  = get_optimal_grid_size(n, block);
        // 注意：一元运算不需要共享内存优化，直接使用简单内核
        unary_kernel<T, ExpOp><<<grid, block, 0, stream>>>(a, c, n, ExpOp{});
    }

    cudaError_t err = cudaGetLastError();
    if (unlikely(err != cudaSuccess))
    {
        THROW_RUNTIME_ERROR("CUDA exp kernel launch failed: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief 运行时类型分发
 * @param dtype 数据类型
 * @param a 输入矩阵A的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param stream CUDA流
 */
void dispatch_exp(DataType dtype, const void *a, void *c, size_t n, cudaStream_t stream = 0)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            launch_exp_kernel<float>(static_cast<const float *>(a), static_cast<float *>(c), n, stream);
            break;
        case DataType::kFloat64:
            launch_exp_kernel<double>(static_cast<const double *>(a), static_cast<double *>(c), n, stream);
            break;
        case DataType::kInt32:
        case DataType::kInt8:
            THROW_INVALID_ARG("Exponential operation not supported for integer types, use float or double");
        default:
            THROW_INVALID_ARG("Unsupported data type for CUDA exp operation: {}", dtype_to_string(dtype));
    }
}

/**
 * @brief exp算子实现
 * @param mat 输入矩阵
 * @return 指数运算结果矩阵
 */
std::unique_ptr<Mat> exp(const OriginMat &mat)
{
    // 使用统一的设备检查
    validation::validate_cuda_device(mat, "exp");

    // 检查数据类型是否支持指数运算 - 使用分支预测优化
    if (unlikely(mat.dtype() != DataType::kFloat32 && mat.dtype() != DataType::kFloat64))
    {
        THROW_INVALID_ARG("Exponential operation only supported for float32 and float64 types");
    }

    // 创建结果张量
    auto result = std::unique_ptr<OriginMat>(new OriginMat(mat.shape(), mat.dtype(), mat.device()));

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    // 启动CUDA内核
    dispatch_exp(mat.dtype(), a_data, c_data, mat.elements());

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
