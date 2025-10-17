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
 * @brief CUDA取负算子实现
 * @details 计算输入张量每个元素的负值
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
void launch_negate_kernel(const T *a, T *c, size_t n, cudaStream_t stream = 0)
{
    // 根据数据大小选择最优的内核
    if (n < 1024)
    {
        // 小数据量：使用简单内核
        dim3 block = get_optimal_block_size(n);
        dim3 grid  = get_optimal_grid_size(n, block);
        unary_kernel<T, NegOp><<<grid, block, 0, stream>>>(a, c, n, NegOp{});
    }
    else if (n < 1024 * 1024)
    {
        // 中等数据量：使用向量化内核
        dim3 block = get_optimal_block_size(n / 4);
        dim3 grid  = get_optimal_grid_size(n / 4, block);
        vectorized_unary_kernel<T, NegOp><<<grid, block, 0, stream>>>(a, c, n, NegOp{});
    }
    else
    {
        // 大数据量：使用简单内核（取负运算不需要共享内存优化）
        dim3 block = get_optimal_block_size(n);
        dim3 grid  = get_optimal_grid_size(n, block);
        unary_kernel<T, NegOp><<<grid, block, 0, stream>>>(a, c, n, NegOp{});
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA negate kernel launch failed: {}", cudaGetErrorString(err));
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
void dispatch_negate(DataType dtype, const void *a, void *c, size_t n, cudaStream_t stream = 0)
{
    device_common::TypeDispatcher::dispatch_void(dtype, [&]<typename T>() {
        launch_negate_kernel<T>(static_cast<const T *>(a), static_cast<T *>(c), n, stream);
    });
}

/**
 * @brief negate算子实现
 * @param mat 输入矩阵
 * @return 取负运算结果矩阵
 */
std::unique_ptr<Mat> negate(const OriginMat &mat)
{
    // 验证输入
    // 使用统一的设备检查
    validation::validate_cuda_device(mat, "negate");

    // 创建结果张量
    auto result = std::unique_ptr<OriginMat>(new OriginMat(mat.shape(), mat.dtype(), mat.device()));

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    // 启动CUDA内核
    dispatch_negate(mat.dtype(), a_data, c_data, mat.elements());

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
