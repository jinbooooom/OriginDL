#ifndef __ORIGIN_DL_CUBLAS_ALGO_H__
#define __ORIGIN_DL_CUBLAS_ALGO_H__

#include "../../basic_types.h"

#ifdef ENABLE_CUBLAS
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#    include "../../../utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief cuBLAS算法优化实现
 * @details 用于优化OriginMat的各种算子计算（如矩阵乘法等）
 *
 * 未来可能扩展的优化算子：
 * - 矩阵乘法 (matmul)
 * - 其他线性代数运算（如矩阵求逆、SVD等）
 */

/**
 * @brief 初始化cuBLAS句柄（线程安全的单例模式）
 * @return cuBLAS句柄
 * @details 自动配置cuBLAS以优化性能：
 * - 启用Tensor Core（如果硬件支持）
 * - 设置数学模式允许FP32->TF32转换（Ampere+）
 * - 使用默认流（异步执行）
 */
cublasHandle_t get_cublas_handle();

/**
 * @brief 释放cuBLAS句柄
 */
void destroy_cublas_handle();

/**
 * @brief 使用cuBLAS优化OriginMat矩阵乘法计算（行主序）
 * @tparam T 数据类型（仅支持float和double）
 * @param a 输入矩阵A（行主序，M×K）
 * @param b 输入矩阵B（行主序，K×N）
 * @param c 输出矩阵C（行主序，M×N）
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 * @details
 * 这是OriginMat矩阵乘法的cuBLAS优化实现。
 * 处理行列主序转换：
 * - 行主序：C = A @ B，其中A是M×K，B是K×N，C是M×N
 * - cuBLAS使用列主序，需要转换为：C' = B' @ A'
 * - 其中A'、B'、C'是列主序矩阵（转置后的行主序矩阵）
 *
 * 注意：cuBLAS仅支持float和double类型，其他类型会编译失败
 */
template <typename T>
void cublas_matmul(const T *a, const T *b, T *c, int M, int N, int K);

// 显式特化声明（仅支持float和double）
template <>
void cublas_matmul<float>(const float *a, const float *b, float *c, int M, int N, int K);

template <>
void cublas_matmul<double>(const double *a, const double *b, double *c, int M, int N, int K);

}  // namespace cuda
}  // namespace origin

#endif  // ENABLE_CUBLAS

#endif  // __ORIGIN_DL_CUBLAS_ALGO_H__
