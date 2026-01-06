#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace origin
{
namespace cuda
{

/**
 * @brief cuBLAS 包装器类
 * @details 管理 cuBLAS handle，提供高性能矩阵乘法接口
 */
class CublasWrapper
{
public:
    /**
     * @brief 获取单例实例
     */
    static CublasWrapper &get_instance();

    /**
     * @brief 获取 cuBLAS handle
     */
    cublasHandle_t get_handle() const { return handle_; }

    /**
     * @brief 执行单精度矩阵乘法: C = alpha * op(A) * op(B) + beta * C
     * @param transa A 的转置操作 (CUBLAS_OP_N 或 CUBLAS_OP_T)
     * @param transb B 的转置操作 (CUBLAS_OP_N 或 CUBLAS_OP_T)
     * @param m A 的行数（转置后）
     * @param n B 的列数（转置后）
     * @param k A 的列数（转置后）= B 的行数（转置后）
     * @param alpha 标量系数
     * @param A 矩阵 A 的数据指针
     * @param lda A 的 leading dimension
     * @param B 矩阵 B 的数据指针
     * @param ldb B 的 leading dimension
     * @param beta 标量系数
     * @param C 矩阵 C 的数据指针（输出）
     * @param ldc C 的 leading dimension
     * @param stream CUDA stream（可选，默认使用默认流）
     */
    void sgemm(cublasOperation_t transa, cublasOperation_t transb,
               int m, int n, int k,
               const float *alpha,
               const float *A, int lda,
               const float *B, int ldb,
               const float *beta,
               float *C, int ldc,
               cudaStream_t stream = nullptr);

private:
    CublasWrapper();
    ~CublasWrapper();
    CublasWrapper(const CublasWrapper &) = delete;
    CublasWrapper &operator=(const CublasWrapper &) = delete;

    cublasHandle_t handle_;
};

}  // namespace cuda
}  // namespace origin

