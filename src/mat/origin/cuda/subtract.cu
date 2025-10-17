#include "origin/mat/origin/cuda/cuda_broadcast.cuh"
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
 * @brief CUDA减法算子实现
 * @details 支持相同形状的张量相减，以及广播运算
 */

/**
 * @brief 类型分发器 - 编译时特化
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param n 元素总数
 * @param stream CUDA流
 */
template <typename T>
void launch_subtract_kernel(const T *a, const T *b, T *c, size_t n, cudaStream_t stream = 0)
{
    // 使用通用的内核启动器
    launch_elementwise_kernel<T, SubtractOp>(a, b, c, n, SubtractOp{}, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA subtract kernel launch failed: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief 运行时类型分发
 * @param dtype 数据类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param stream CUDA流
 */
void dispatch_subtract(DataType dtype, const void *a, const void *b, void *c, size_t n, cudaStream_t stream = 0)
{
    device_common::TypeDispatcher::dispatch_void(dtype, [&]<typename T>() {
        launch_subtract_kernel<T>(static_cast<const T *>(a), static_cast<const T *>(b), static_cast<T *>(c), n, stream);
    });
}

/**
 * @brief 简单广播减法内核启动
 * @param dtype 数据类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param a_elements A矩阵元素数量
 * @param b_elements B矩阵元素数量
 * @param c_elements C矩阵元素数量
 * @param stream CUDA流
 */
void dispatch_scalar_broadcast_subtract(DataType dtype,
                                        const void *a,
                                        const void *b,
                                        void *c,
                                        size_t a_elements,
                                        size_t b_elements,
                                        size_t c_elements,
                                        cudaStream_t stream = 0)
{
    dim3 block = get_optimal_block_size(c_elements);
    dim3 grid  = get_optimal_grid_size(c_elements, block);

    device_common::TypeDispatcher::dispatch_void(dtype, [&]<typename T>() {
        scalar_broadcast_kernel<T, SubtractOp>
            <<<grid, block, 0, stream>>>(static_cast<const T *>(a), static_cast<const T *>(b), static_cast<T *>(c),
                                         a_elements, b_elements, c_elements, SubtractOp{});
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA scalar broadcast subtract kernel launch failed: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief subtract算子实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 减法结果矩阵
 */
std::unique_ptr<Mat> subtract(const OriginMat &a, const OriginMat &b)
{
    // 验证输入
    if (a.dtype() != b.dtype())
    {
        THROW_INVALID_ARG("Data type mismatch in CUDA subtract: {} vs {}", dtype_to_string(a.dtype()),
                          dtype_to_string(b.dtype()));
    }

    // 使用统一的设备检查
    validation::validate_same_device(a, b, "subtract");
    validation::validate_cuda_device(a, "subtract");

    // 计算广播形状
    Shape result_shape = compute_broadcast_shape(a, b);
    auto result        = std::unique_ptr<OriginMat>(new OriginMat(result_shape, a.dtype(), a.device()));

    // 获取数据指针
    const void *a_data = a.storage()->data();
    const void *b_data = b.storage()->data();
    void *c_data       = result->storage()->data();

    // 检查是否是简单广播
    if (a.elements() == 1 || b.elements() == 1)
    {
        // 简单广播：一个操作数是标量
        dispatch_scalar_broadcast_subtract(a.dtype(), a_data, b_data, c_data, a.elements(), b.elements(),
                                           result->elements());
    }
    else if (a.shape() == b.shape())
    {
        // 相同形状：直接元素级运算
        dispatch_subtract(a.dtype(), a_data, b_data, c_data, a.elements());
    }
    else
    {
        // 复杂广播：需要计算步长信息
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CUDA subtract operation");
    }

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
