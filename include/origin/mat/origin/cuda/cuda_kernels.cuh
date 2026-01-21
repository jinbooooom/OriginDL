#ifndef __ORIGIN_DL_CUDA_KERNELS_H__
#define __ORIGIN_DL_CUDA_KERNELS_H__

#include <cuda_runtime.h>
#include "../../basic_types.h"
#include "../device_common/operation_templates.h"

namespace origin
{
namespace cuda
{

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
                                   Op op);

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
__global__ void unary_kernel(const T *__restrict__ a, T *__restrict__ c, size_t n, Op op);

/**
 * @brief 类型转换内核
 * @tparam SrcT 源数据类型
 * @tparam DstT 目标数据类型
 * @param src 输入矩阵的设备指针
 * @param dst 输出矩阵的设备指针
 * @param n 元素总数
 */
template <typename SrcT, typename DstT>
__global__ void type_conversion_kernel(const SrcT *__restrict__ src, DstT *__restrict__ dst, size_t n);

/**
 * @brief 索引写入内核（单个元素）
 * @tparam T 数据类型
 * @param data 数据指针
 * @param index 线性索引
 * @param value 要写入的值
 */
template <typename T>
__global__ void index_put_kernel(T *data, size_t index, T value);

/**
 * @brief 启动索引写入内核（单个元素）
 * @tparam T 数据类型
 * @param data 数据指针
 * @param index 线性索引
 * @param value 要写入的值
 * @param stream CUDA流
 */
template <typename T>
void launch_index_put_kernel(T *data, size_t index, T value, cudaStream_t stream = 0);

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
__global__ void scalar_kernel(const T *__restrict__ a, T scalar, T *__restrict__ c, size_t n, Op op);

// ============================================================================
// 向量化优化内核
// ============================================================================

/**
 * @brief 向量化二元运算内核（使用float4/double4）
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 * @details 一次处理4个元素，提高内存带宽利用率
 */
template <typename T, typename Op>
__global__ void vectorized_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c, size_t n, Op op);

/**
 * @brief 向量化一元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 */
template <typename T, typename Op>
__global__ void vectorized_unary_kernel(const T *__restrict__ a, T *__restrict__ c, size_t n, Op op);

// ============================================================================
// 共享内存优化内核
// ============================================================================

/**
 * @brief 使用共享内存的二元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param op 操作函数对象
 * @details 使用共享内存减少全局内存访问，适合重复访问的数据
 */
template <typename T, typename Op>
__global__ void shared_memory_kernel(const T *__restrict__ a,
                                     const T *__restrict__ b,
                                     T *__restrict__ c,
                                     size_t n,
                                     Op op);

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
                                        Op op);

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
                                         Op op);

// ============================================================================
// 矩阵运算内核
// ============================================================================

/**
 * @brief 基础矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param M A矩阵行数
 * @param N B矩阵列数
 * @param K A矩阵列数/B矩阵行数
 */
template <typename T>
__global__ void matmul_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c, int M, int N, int K);

/**
 * @brief 分块矩阵乘法内核（使用共享内存优化）
 * @tparam T 数据类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param M A矩阵行数
 * @param N B矩阵列数
 * @param K A矩阵列数/B矩阵行数
 * @details 使用共享内存减少全局内存访问，提高性能
 */
template <typename T>
__global__ void matmul_tiled_kernel(const T *__restrict__ a,
                                    const T *__restrict__ b,
                                    T *__restrict__ c,
                                    int M,
                                    int N,
                                    int K);

// ============================================================================
// 归约运算内核
// ============================================================================

/**
 * @brief 轴求和内核
 * @tparam T 数据类型
 * @param input 输入数据设备指针
 * @param output 输出数据设备指针
 * @param input_shape 输入形状数组
 * @param output_shape 输出形状数组
 * @param axis 求和轴
 * @param ndims 维度数量
 * @param input_elements 输入元素总数
 * @param output_elements 输出元素总数
 */
template <typename T>
__global__ void axis_sum_kernel(const T *__restrict__ input,
                                T *__restrict__ output,
                                const int *input_shape,
                                const int *output_shape,
                                int axis,
                                int ndims,
                                size_t input_elements,
                                size_t output_elements);

/**
 * @brief 全元素求和内核（使用树状归约）
 * @tparam T 数据类型
 * @param input 输入数据设备指针
 * @param output 输出数据设备指针（部分和）
 * @param n 元素总数
 * @details 使用树状归约算法，高效计算所有元素的和
 */
template <typename T>
__global__ void sum_all_kernel(const T *__restrict__ input, T *__restrict__ output, size_t n);

// ============================================================================
// 形状操作内核
// ============================================================================

/**
 * @brief 转置内核
 * @tparam T 数据类型
 * @param input 输入数据设备指针
 * @param output 输出数据设备指针
 * @param rows 行数
 * @param cols 列数
 */
template <typename T>
__global__ void transpose_kernel(const T *__restrict__ input, T *__restrict__ output, int rows, int cols);

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
void launch_elementwise_kernel(const T *a, const T *b, T *c, size_t n, Op op, cudaStream_t stream = 0);

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
void launch_unary_kernel(const T *a, T *c, size_t n, Op op, cudaStream_t stream = 0);

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
void launch_type_conversion_kernel(const SrcT *src, DstT *dst, size_t n, cudaStream_t stream = 0);

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
void launch_scalar_kernel(const T *a, T scalar, T *c, size_t n, Op op, cudaStream_t stream = 0);

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
                                    cudaStream_t stream = 0);

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
                                     cudaStream_t stream = 0);

/**
 * @brief 启动矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A的设备指针
 * @param b 输入矩阵B的设备指针
 * @param c 输出矩阵C的设备指针
 * @param M A矩阵行数
 * @param N B矩阵列数
 * @param K A矩阵列数/B矩阵行数
 * @param stream CUDA流
 */
template <typename T>
void launch_matmul_kernel(const T *a, const T *b, T *c, int M, int N, int K, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_KERNELS_H__
