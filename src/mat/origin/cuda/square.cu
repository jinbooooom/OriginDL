#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA平方算子实现
 * @details 计算输入张量每个元素的平方值
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
void launch_square_kernel(const T *a, T *c, size_t n, cudaStream_t stream = 0)
{
    // 根据数据大小选择最优的内核
    if (n < 1024)
    {
        // 小数据量：使用简单内核
        dim3 block = get_optimal_block_size(n);
        dim3 grid  = get_optimal_grid_size(n, block);
        unary_kernel<T, SquareOp><<<grid, block, 0, stream>>>(a, c, n, SquareOp{});
    }
    else if (n < 1024 * 1024)
    {
        // 中等数据量：使用向量化内核
        dim3 block = get_optimal_block_size(n / 4);
        dim3 grid  = get_optimal_grid_size(n / 4, block);
        vectorized_unary_kernel<T, SquareOp><<<grid, block, 0, stream>>>(a, c, n, SquareOp{});
    }
    else
    {
        // 大数据量：使用简单内核（平方运算不需要共享内存优化）
        dim3 block = get_optimal_block_size(n);
        dim3 grid  = get_optimal_grid_size(n, block);
        unary_kernel<T, SquareOp><<<grid, block, 0, stream>>>(a, c, n, SquareOp{});
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA square kernel launch failed: {}", cudaGetErrorString(err));
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
void dispatch_square(DataType dtype, const void *a, void *c, size_t n, cudaStream_t stream = 0)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            launch_square_kernel<float>(static_cast<const float *>(a), static_cast<float *>(c), n, stream);
            break;
        case DataType::kFloat64:
            launch_square_kernel<double>(static_cast<const double *>(a), static_cast<double *>(c), n, stream);
            break;
        case DataType::kInt32:
            launch_square_kernel<int32_t>(static_cast<const int32_t *>(a), static_cast<int32_t *>(c), n, stream);
            break;
        case DataType::kInt8:
            launch_square_kernel<int8_t>(static_cast<const int8_t *>(a), static_cast<int8_t *>(c), n, stream);
            break;
        default:
            THROW_INVALID_ARG("Unsupported data type for CUDA square operation: {}", dtype_to_string(dtype));
    }
}

/**
 * @brief square算子实现
 * @param mat 输入矩阵
 * @return 平方运算结果矩阵
 */
std::unique_ptr<Mat> square(const OriginMat &mat)
{
    // 验证输入
    // 使用统一的设备检查
    validation::validate_cuda_device(mat, "square");

    // 创建结果张量
    auto result = std::unique_ptr<OriginMat>(new OriginMat(mat.shape(), mat.dtype(), mat.device()));

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    // 启动CUDA内核
    dispatch_square(mat.dtype(), a_data, c_data, mat.elements());

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
