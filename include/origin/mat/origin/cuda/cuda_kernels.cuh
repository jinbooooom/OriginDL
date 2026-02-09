#ifndef __ORIGIN_DL_CUDA_KERNELS_H__
#define __ORIGIN_DL_CUDA_KERNELS_H__

#include <cuda_runtime.h>
#include "../../basic_types.h"
#include "../device_common/operation_templates.h"

#ifdef __CUDACC__
#    include "cuda_utils.cuh"
#endif

#ifdef WITH_CUDA
#    include "cuda_utils.cuh"
#endif

namespace origin
{
namespace cuda
{

#ifdef WITH_CUDA
// launch_index_put_kernel 和 launch_clone_kernel 在 origin_mat.cpp（普通 C++ 文件）中被调用
// 其他启动函数（如 launch_elementwise_kernel）只在 .cu 文件中被调用
// 为了保证编译通过，所以前向声明一下。
template <typename T>
void launch_index_put_kernel(T *data, size_t index, T value, cudaStream_t stream = 0);

template <typename T>
void launch_clone_kernel(const T *src,
                         T *dst,
                         const size_t *shape,
                         const size_t *src_strides,
                         const size_t *output_strides,
                         size_t ndim,
                         size_t total_elements,
                         cudaStream_t stream = 0);
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
__global__ void elementwise_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c, size_t n, Op op)
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

/**
 * @brief CUDA clone kernel：按逻辑顺序拷贝非连续张量
 * @tparam T 数据类型
 * @param src 源数据指针 
 * @param dst 目标数据指针（连续存储）
 * @param shape 源张量的形状（与 src_strides 对应）
 * @param src_strides 源张量的 strides（可能非连续）
 * @param output_strides 输出张量的连续 strides（用于计算逻辑索引，对应源张量的 shape）
 * @param ndim 维度数
 * @param total_elements 元素总数
 *
 * ============================================================================
 * 算法原理
 * ============================================================================
 *
 * 该 kernel 的核心目标是将非连续张量按逻辑顺序拷贝到连续内存中。
 * 关键思想是：将输出位置的线性索引转换为源张量的物理内存偏移。
 *
 * 算法分为三个步骤：
 *
 * 1. 线性索引 -> 多维坐标
 *    使用输出张量的连续 strides，将线性索引 idx 转换为多维坐标 (i, j, k, ...)
 *    这些坐标对应源张量的逻辑位置
 *
 * 2. 多维坐标 -> 源张量物理偏移
 *    使用相同的坐标和源张量的 strides 计算源张量中的物理内存偏移
 *
 * 3. 拷贝元素
 *    从源张量的物理位置读取，写入到目标张量的连续位置
 *
 * ============================================================================
 * 步骤1详解：线性索引 -> 多维坐标
 * ============================================================================
 *
 * 这是算法的核心步骤，需要理解 strides 的含义和计算方法。
 *
 * 1.1 Strides 的含义
 *    对于形状为 [d0, d1, d2, ..., d(n-1)] 的张量，连续 strides 的计算方式：
 *        strides[n-1] = 1
 *        strides[i] = strides[i+1] * d(i+1)  (从后往前计算)
 *
 *    例如：shape = [3, 2]
 *        strides[1] = 1
 *        strides[0] = strides[1] * 2 = 1 * 2 = 2
 *        所以 output_strides = [2, 1]
 *
 * 1.2 线性索引到多维坐标的转换原理
 *    对于连续存储的张量，线性索引 idx 与多维坐标 (i0, i1, i2, ...) 的关系：
 *        idx = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + ...
 *
 *    要从 idx 反推出坐标，需要"反向"计算：
 *        i0 = idx / strides[0]           (整数除法)
 *        remainder = idx % strides[0]    (余数)
 *        i1 = remainder / strides[1]
 *        remainder = remainder % strides[1]
 *        i2 = remainder / strides[2]
 *        ...
 *
 * 1.3 算法实现
 *    代码中的实现：
 *        remaining = idx
 *        for d = 0 to ndim-1:
 *            coords[d] = remaining / output_strides[d]
 *            remaining = remaining % output_strides[d]
 *
 * 1.4 具体示例（shape = [3, 2], output_strides = [2, 1]）
 *
 *    idx = 0:
 *        d=0: coords[0] = 0 / 2 = 0, remaining = 0 % 2 = 0
 *        d=1: coords[1] = 0 / 1 = 0, remaining = 0 % 1 = 0
 *        结果: coords = [0, 0] -> 位置 (0, 0)
 *
 *    idx = 1:
 *        d=0: coords[0] = 1 / 2 = 0, remaining = 1 % 2 = 1
 *        d=1: coords[1] = 1 / 1 = 1, remaining = 1 % 1 = 0
 *        结果: coords = [0, 1] -> 位置 (0, 1)
 *
 *    idx = 2:
 *        d=0: coords[0] = 2 / 2 = 1, remaining = 2 % 2 = 0
 *        d=1: coords[1] = 0 / 1 = 0, remaining = 0 % 1 = 0
 *        结果: coords = [1, 0] -> 位置 (1, 0)
 *
 *    idx = 3:
 *        d=0: coords[0] = 3 / 2 = 1, remaining = 3 % 2 = 1
 *        d=1: coords[1] = 1 / 1 = 1, remaining = 1 % 1 = 0
 *        结果: coords = [1, 1] -> 位置 (1, 1)
 *
 *    idx = 4:
 *        d=0: coords[0] = 4 / 2 = 2, remaining = 4 % 2 = 0
 *        d=1: coords[1] = 0 / 1 = 0, remaining = 0 % 1 = 0
 *        结果: coords = [2, 0] -> 位置 (2, 0)
 *
 *    idx = 5:
 *        d=0: coords[0] = 5 / 2 = 2, remaining = 5 % 2 = 1
 *        d=1: coords[1] = 1 / 1 = 1, remaining = 1 % 1 = 0
 *        结果: coords = [2, 1] -> 位置 (2, 1)
 *
 * 1.5 为什么使用 output_strides 而不是 shape？
 *    因为 strides 包含了维度大小的信息，可以直接用于计算坐标。
 *    使用 strides 的好处是：
 *    - 直接反映了内存布局
 *    - 计算效率高（只需要除法和取模）
 *    - 适用于任意维度的张量
 *
 * 1.6 三维示例（shape = [2, 2, 2], output_strides = [4, 2, 1]）
 *
 *    idx = 0:  coords = [0, 0, 0] -> (0, 0, 0)
 *    idx = 1:  coords = [0, 0, 1] -> (0, 0, 1)
 *    idx = 2:  coords = [0, 1, 0] -> (0, 1, 0)
 *    idx = 3:  coords = [0, 1, 1] -> (0, 1, 1)
 *    idx = 4:  coords = [1, 0, 0] -> (1, 0, 0)
 *    idx = 5:  coords = [1, 0, 1] -> (1, 0, 1)
 *    idx = 6:  coords = [1, 1, 0] -> (1, 1, 0)
 *    idx = 7:  coords = [1, 1, 1] -> (1, 1, 1)
 *
 *    验证：idx = 5 = 1*4 + 0*2 + 1*1 = 4 + 0 + 1 = 5 
 *
 * ============================================================================
 * 完整示例：转置后 reshape 的场景
 * ============================================================================
 *
 * 假设有一个 2×3 的张量经过转置，然后需要 reshape：
 *
 * 步骤1：原始张量
 *   原始数据（内存顺序）: [1, 2, 3, 4, 5, 6]
 *   原始形状: [2, 3]
 *   原始逻辑顺序: [[1, 2, 3], [4, 5, 6]]
 *
 * 步骤2：转置操作
 *   转置后形状: [3, 2]  <- 这是源张量的 shape（传入 kernel）
 *   转置后逻辑顺序: [[1, 4], [2, 5], [3, 6]]
 *   转置后内存顺序: [1, 2, 3, 4, 5, 6] (未变，但strides变了)
 *   转置后strides: [1, 3]  <- 这是 src_strides（非连续）
 *
 * 步骤3：clone_kernel 拷贝过程（将非连续张量转为连续）
 *   输出形状: [3, 2]  <- 与源张量相同（因为 contiguous 保持形状）
 *   output_strides: [2, 1]  <- 连续 strides
 *
 *   拷贝过程：
 *
 *   | idx | coords | src_offset计算 | 逻辑值 | 物理位置 | 拷贝到 dst[idx] |
 *   |-----|--------|----------------|--------|----------|----------------|
 *   | 0   | (0,0)  | 0*1+0*3=0      | 1      | src[0]   | dst[0] = 1     |
 *   | 1   | (0,1)  | 0*1+1*3=3      | 4      | src[3]   | dst[1] = 4     |
 *   | 2   | (1,0)  | 1*1+0*3=1      | 2      | src[1]   | dst[2] = 2     |
 *   | 3   | (1,1)  | 1*1+1*3=4      | 5      | src[4]   | dst[3] = 5     |
 *   | 4   | (2,0)  | 2*1+0*3=2      | 3      | src[2]   | dst[4] = 3     |
 *   | 5   | (2,1)  | 2*1+1*3=5      | 6      | src[5]   | dst[5] = 6     |
 *
 *   结果：dst = [1, 4, 2, 5, 3, 6]（按逻辑顺序，连续存储）
 *
 * 步骤4：reshape 操作（在 contiguous 之后）
 *   对连续副本进行 reshape，例如 reshape 为 [6]
 *   结果: [1, 4, 2, 5, 3, 6]（零拷贝，只是改变视图）
 *
 */
template <typename T>
__global__ void clone_kernel(const T *__restrict__ src,
                             T *__restrict__ dst,
                             const size_t *shape,
                             const size_t *src_strides,
                             const size_t *output_strides,
                             size_t ndim,
                             size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        // 步骤1：将线性索引转换为多维坐标（按输出形状）
        // 使用输出张量的连续 strides，将线性索引 idx 转换为多维坐标
        size_t coords[8];  // 支持最多8维
        size_t remaining = idx;
        for (size_t d = 0; d < ndim; ++d)
        {
            coords[d] = remaining / output_strides[d];
            remaining %= output_strides[d];
        }

        // 步骤2：计算源张量的物理偏移（使用实际的 strides）
        // 使用相同的坐标和源张量的 strides 计算源张量中的物理内存偏移
        size_t src_offset = 0;
        for (size_t d = 0; d < ndim; ++d)
        {
            src_offset += coords[d] * src_strides[d];
        }

        // 步骤3：拷贝元素（目标位置是连续的，所以直接使用 idx）
        // 从源张量的物理位置读取，写入到目标张量的连续位置
        dst[idx] = src[src_offset];
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

/**
 * @brief 启动 clone kernel：按逻辑顺序拷贝非连续张量
 * @tparam T 数据类型
 * @param src 源数据指针
 * @param dst 目标数据指针（连续存储）
 * @param shape 张量形状
 * @param src_strides 源张量的 strides
 * @param output_strides 输出张量的连续 strides
 * @param ndim 维度数
 * @param total_elements 元素总数
 * @param stream CUDA流
 * @note 实现在 cuda_kernels.cu 中，用 nvcc 编译
 */
template <typename T>
void launch_clone_kernel(const T *src,
                         T *dst,
                         const size_t *shape,
                         const size_t *src_strides,
                         const size_t *output_strides,
                         size_t ndim,
                         size_t total_elements,
                         cudaStream_t stream = 0);
#endif  // __CUDACC__

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_KERNELS_H__
