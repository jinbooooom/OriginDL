#ifndef __ORIGIN_DL_CUDA_KERNELS_H__
#define __ORIGIN_DL_CUDA_KERNELS_H__

#include <cuda_runtime.h>
#include "../../basic_types.h"
#include "../device_common/operation_templates.h"

#ifdef __CUDACC__
#include "cuda_utils.cuh"
#endif

#ifdef WITH_CUDA
#include "cuda_utils.cuh"
#endif

namespace origin
{
namespace cuda
{

#ifdef WITH_CUDA
// launch_index_put_kernel 在 origin_mat.cpp（普通 C++ 文件）中被调用
// 其他启动函数（如 launch_elementwise_kernel）只在 .cu 文件中被调用
// 为了保证编译通过，所以前向声明一下。
template <typename T>
void launch_index_put_kernel(T *data, size_t index, T value, cudaStream_t stream = 0);
#endif  // WITH_CUDA

#ifdef __CUDACC__
/**
 * @brief CUDA内核函数集合
 * @details 包含所有高性能CUDA内核实现，支持多种数据类型和优化策略
 */

// ============================================================================
// 基础元素级运算内核
// ============================================================================

/**
 * @brief 基础元素级二元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 * @details 使用合并内存访问模式，最大化内存带宽利用率
 */
template <typename T, typename Op>
__global__ void elementwise_kernel(const T *__restrict__ a,
                                   const T *__restrict__ b,
                                   T *__restrict__ c,
                                   size_t n,
                                   Op op)
{
    // 计算全局线程索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，确保不越界
    if (idx < n)
    {
        // 执行元素级操作
        // 使用__restrict__确保编译器知道指针不会重叠，允许更激进的优化
        c[idx] = op(a[idx], b[idx]);
    }
}

/**
 * @brief 基础元素级一元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 */
template <typename T, typename Op>
__global__ void unary_kernel(const T *__restrict__ a, T *__restrict__ c, size_t n, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = op(a[idx]);
    }
}

/**
 * @brief 类型转换内核
 * @tparam SrcT 源数据类型
 * @tparam DstT 目标数据类型
 * @param src 输入矩阵的设备指针
 * @param dst 输出矩阵的设备指针
 * @param n 元素总数
 * @details 在GPU上直接进行类型转换，避免CPU-CUDA数据传输
 */
template <typename SrcT, typename DstT>
__global__ void type_conversion_kernel(const SrcT *__restrict__ src, DstT *__restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        dst[idx] = static_cast<DstT>(src[idx]);
    }
}

/**
 * @brief 索引写入内核（单个元素）
 * @tparam T 数据类型
 * @param data 数据指针
 * @param index 线性索引
 * @param value 要写入的值
 */
template <typename T>
__global__ void index_put_kernel(T *data, size_t index, T value)
{
    // 只写入指定索引位置的值
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        data[index] = value;
    }
}


/**
 * @brief 标量运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param scalar 标量值
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 */
template <typename T, typename Op>
__global__ void scalar_kernel(const T *__restrict__ a, T scalar, T *__restrict__ c, size_t n, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = op(a[idx], scalar);
    }
}

// ============================================================================
// 广播运算内核
// ============================================================================

/**
 * @brief 简单广播内核（一个操作数是标量）
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param a_elements A矩阵元素数量
 * @param b_elements B矩阵元素数量
 * @param c_elements C矩阵元素数量
 * @param op 操作函数对象
 * @details 用于处理简单广播情况，其中一个操作数是标量（只有1个元素）。
 *          这是广播运算中最常见的情况，性能最优。
 * @note 此内核假设其中一个张量是标量，另一个张量是普通张量。
 *       如果两个张量都是标量，此内核仍然可以正常工作。
 */
template <typename T, typename Op>
__global__ void simple_broadcast_kernel(const T *__restrict__ a,
                                        const T *__restrict__ b,
                                        T *__restrict__ c,
                                        size_t a_elements,
                                        size_t b_elements,
                                        size_t c_elements,
                                        Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < c_elements)
    {
        // 计算源索引：如果操作数是标量，索引为0，否则为当前索引
        size_t a_idx = (a_elements == 1) ? 0 : idx;
        size_t b_idx = (b_elements == 1) ? 0 : idx;

        c[idx] = op(a[a_idx], b[b_idx]);
    }
}

/**
 * @brief 复杂广播内核（处理不同维度的张量）
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param a_strides A矩阵的步长数组
 * @param b_strides B矩阵的步长数组
 * @param c_strides C矩阵的步长数组
 * @param a_shape A矩阵的形状数组
 * @param b_shape B矩阵的形状数组
 * @param c_shape C矩阵的形状数组
 * @param ndims 维度数量
 * @param total_elements 总元素数量
 * @param op 操作函数对象
 */
template <typename T, typename Op>
__global__ void complex_broadcast_kernel(const T *__restrict__ a,
                                         const T *__restrict__ b,
                                         T *__restrict__ c,
                                         const int *a_strides,
                                         const int *b_strides,
                                         const int *c_strides,
                                         const int *a_shape,
                                         const int *b_shape,
                                         const int *c_shape,
                                         int ndims,
                                         size_t total_elements,
                                         Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        // 将线性索引转换为多维索引
        int c_indices[8];  // 假设最大8维
        int temp = idx;
        for (int i = ndims - 1; i >= 0; --i)
        {
            c_indices[i] = temp % c_shape[i];
            temp /= c_shape[i];
        }

        // 计算a和b的线性索引
        int a_idx = 0, b_idx = 0;
        for (int i = 0; i < ndims; ++i)
        {
            // 如果维度大小为1，则索引为0（广播）
            int a_dim_idx = (a_shape[i] == 1) ? 0 : c_indices[i];
            int b_dim_idx = (b_shape[i] == 1) ? 0 : c_indices[i];

            a_idx += a_dim_idx * a_strides[i];
            b_idx += b_dim_idx * b_strides[i];
        }

        c[idx] = op(a[a_idx], b[b_idx]);
    }
}

// ============================================================================
// 内核启动函数
// ============================================================================

/**
 * @brief 启动元素级二元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 * @param stream CUDA流
 */
template <typename T, typename Op>
void launch_elementwise_kernel(const T *a, const T *b, T *c, size_t n, Op op, cudaStream_t stream = 0)
{
    // 根据数据大小选择最优的线程块大小
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    // 启动内核
    elementwise_kernel<T, Op><<<grid, block, 0, stream>>>(a, b, c, n, op);
}

/**
 * @brief 启动元素级一元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 * @param stream CUDA流
 */
template <typename T, typename Op>
void launch_unary_kernel(const T *a, T *c, size_t n, Op op, cudaStream_t stream = 0)
{
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    unary_kernel<T, Op><<<grid, block, 0, stream>>>(a, c, n, op);
}

/**
 * @brief 启动类型转换内核
 * @tparam SrcT 源数据类型
 * @tparam DstT 目标数据类型
 * @param src 输入矩阵的设备指针
 * @param dst 输出矩阵的设备指针
 * @param n 元素总数
 * @param stream CUDA流
 */
template <typename SrcT, typename DstT>
void launch_type_conversion_kernel(const SrcT *src, DstT *dst, size_t n, cudaStream_t stream = 0)
{
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    type_conversion_kernel<SrcT, DstT><<<grid, block, 0, stream>>>(src, dst, n);
}

/**
 * @brief 启动标量运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param scalar 标量值
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 * @param stream CUDA流
 */
template <typename T, typename Op>
void launch_scalar_kernel(const T *a, T scalar, T *c, size_t n, Op op, cudaStream_t stream = 0)
{
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    scalar_kernel<T, Op><<<grid, block, 0, stream>>>(a, scalar, c, n, op);
}

/**
 * @brief 启动简单广播内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param a_elements A矩阵元素数量
 * @param b_elements B矩阵元素数量
 * @param c_elements C矩阵元素数量
 * @param op 操作函数对象
 * @param stream CUDA流
 * @details 用于启动简单广播运算，其中一个操作数是标量。
 *          自动计算最优的网格和块大小。
 */
template <typename T, typename Op>
void launch_simple_broadcast_kernel(const T *a,
                                    const T *b,
                                    T *c,
                                    size_t a_elements,
                                    size_t b_elements,
                                    size_t c_elements,
                                    Op op,
                                    cudaStream_t stream = 0)
{
    dim3 block = get_optimal_block_size(c_elements);
    dim3 grid  = get_optimal_grid_size(c_elements, block);

    simple_broadcast_kernel<T, Op><<<grid, block, 0, stream>>>(a, b, c, a_elements, b_elements, c_elements, op);
}

/**
 * @brief 启动复杂广播内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param a_strides A矩阵的步长数组
 * @param b_strides B矩阵的步长数组
 * @param c_strides C矩阵的步长数组
 * @param a_shape A矩阵的形状数组
 * @param b_shape B矩阵的形状数组
 * @param c_shape C矩阵的形状数组
 * @param ndims 维度数量
 * @param total_elements 总元素数量
 * @param op 操作函数对象
 * @param stream CUDA流
 * @details 用于启动复杂广播运算，处理不同维度的张量之间的广播。
 *          需要预先计算步长和形状信息，适用于任意维度的广播。
 * @note 此函数需要调用者确保步长和形状数组在设备内存中，且维度信息正确。
 *       建议使用 compute_broadcast_strides 等辅助函数来计算这些参数。
 */
template <typename T, typename Op>
void launch_complex_broadcast_kernel(const T *a,
                                     const T *b,
                                     T *c,
                                     const int *a_strides,
                                     const int *b_strides,
                                     const int *c_strides,
                                     const int *a_shape,
                                     const int *b_shape,
                                     const int *c_shape,
                                     int ndims,
                                     size_t total_elements,
                                     Op op,
                                     cudaStream_t stream = 0)
{
    dim3 block = get_optimal_block_size(total_elements);
    dim3 grid  = get_optimal_grid_size(total_elements, block);

    complex_broadcast_kernel<T, Op><<<grid, block, 0, stream>>>(a, b, c, a_strides, b_strides, c_strides, a_shape,
                                                                b_shape, c_shape, ndims, total_elements, op);
}

/**
 * @brief 启动索引写入内核（单个元素）
 * @tparam T 数据类型
 * @param data 数据指针
 * @param index 线性索引
 * @param value 要写入的值
 * @param stream CUDA流
 * @note 实现在 cuda_kernels.cu 中，用 nvcc 编译
 */
template <typename T>
void launch_index_put_kernel(T *data, size_t index, T value, cudaStream_t stream = 0);
#endif  // __CUDACC__

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_KERNELS_H__
