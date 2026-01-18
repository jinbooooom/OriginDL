#ifdef ENABLE_CUBLAS

#    include <cuda_runtime.h>
#    include <mutex>
#    include "origin/mat/origin/cuda/cublas_algo.cuh"

namespace origin
{
namespace cuda
{

// cuBLAS句柄（单例模式）
static cublasHandle_t s_cublas_handle = nullptr;
static std::mutex s_cublas_mutex;

/**
 * @brief 初始化cuBLAS句柄（线程安全的单例模式）
 * @details 配置cuBLAS以优化性能：
 * - 启用Tensor Core（如果硬件支持）
 * - 设置数学模式允许FP32->TF32转换（Ampere+）
 * - 使用默认流（异步执行）
 */
cublasHandle_t get_cublas_handle()
{
    if (s_cublas_handle == nullptr)
    {
        std::lock_guard<std::mutex> lock(s_cublas_mutex);
        if (s_cublas_handle == nullptr)
        {
            cublasStatus_t status = cublasCreate(&s_cublas_handle);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                THROW_RUNTIME_ERROR("cuBLAS initialization failed: {}", static_cast<int>(status));
            }

            // 设置数学模式，允许Tensor Core和TF32（如果硬件支持）
            // 通过运行时检查cuBLAS版本来设置数学模式
            int cublas_version = 0;
            cublasGetVersion(s_cublas_handle, &cublas_version);

            // cuBLAS版本格式：major*1000 + minor*100
            int major = cublas_version / 1000;

#    if defined(CUBLAS_DEFAULT_MATH)
            if (major >= 11)
            {
                // cuBLAS 11.0+ 支持数学模式设置
                // cuBLAS 11.1+ 支持 CUBLAS_DEFAULT_MATH（自动选择最优模式）
                int minor = (cublas_version % 1000) / 100;
                if (major > 11 || (major == 11 && minor >= 1))
                {
                    cublasSetMathMode(s_cublas_handle, CUBLAS_DEFAULT_MATH);
                }
            }
#    elif defined(CUBLAS_TF32_TENSOR_OP_MATH)
            if (major == 11)
            {
                // cuBLAS 11.0 使用 CUBLAS_TF32_TENSOR_OP_MATH
                int minor = (cublas_version % 1000) / 100;
                if (minor == 0)
                {
                    cublasSetMathMode(s_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
                }
            }
#    elif defined(CUBLAS_TENSOR_OP_MATH)
            if (major >= 10 && major < 11)
            {
                // cuBLAS 10.0+ 支持 CUBLAS_TENSOR_OP_MATH
                cublasSetMathMode(s_cublas_handle, CUBLAS_TENSOR_OP_MATH);
            }
#    endif
            // 如果cuBLAS版本较老或宏未定义，使用默认模式（不设置数学模式）

            // 注意：不设置stream，使用默认流（NULL stream）
            // cuBLAS会自动与CUDA runtime同步，避免不必要的同步开销
        }
    }
    return s_cublas_handle;
}

/**
 * @brief 释放cuBLAS句柄
 */
void destroy_cublas_handle()
{
    std::lock_guard<std::mutex> lock(s_cublas_mutex);
    if (s_cublas_handle != nullptr)
    {
        cublasDestroy(s_cublas_handle);
        s_cublas_handle = nullptr;
    }
}

/**
 * @brief 使用cuBLAS优化OriginMat矩阵乘法计算（行主序）
 * @details
 * cuBLAS算法优化：矩阵乘法实现。
 * 行列主序转换说明：
 * - 行主序：C = A @ B，其中A是M×K，B是K×N，C是M×N
 * - cuBLAS使用列主序，要计算行主序的 C = A @ B
 * - 对于行主序数据，我们把它们当作列主序的转置：
 *   - 行主序 A (M×K) 在列主序中会被解释为 A^T (K×M)
 *   - 行主序 B (K×N) 在列主序中会被解释为 B^T (N×K)
 * - 要计算 C = A @ B，在列主序中计算 C^T = B^T @ A^T
 * - 使用 CUBLAS_OP_N，交换 M 和 N，以及 A 和 B 的位置
 * - leading dimension 使用行主序的列数（即矩阵的列数）
 *
 * 参考：https://docs.nvidia.com/cuda/cublas/index.html
 */
template <>
void cublas_matmul<float>(const float *a, const float *b, float *c, int M, int N, int K)
{
    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS使用列主序，要计算行主序的 C = A @ B
    // 行主序：A (M×K), B (K×N), C (M×N)
    // 在列主序中，行主序的A被视为A^T (K×M)，B被视为B^T (N×K)
    // 要计算 C = A @ B，在列主序中计算 C^T = B^T @ A^T
    // 参数顺序：B, A, C（注意顺序）
    // 维度：m=N, n=M, k=K（交换M和N）

    // 使用cublasSgemm进行矩阵乘法
    // 对于大矩阵，cuBLAS会自动选择最优算法（包括Tensor Core）
    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N,  // transa: 不转置B^T，使用B^T (N×K)
                                        CUBLAS_OP_N,  // transb: 不转置A^T，使用A^T (K×M)
                                        N,            // m: C^T的行数 = B^T的行数 = N
                                        M,            // n: C^T的列数 = A^T的列数 = M
                                        K,            // k: 公共维度 = B^T的列数 = A^T的行数 = K
                                        &alpha,       // alpha
                                        b,            // B (K×N行主序)，在列主序中是B^T (N×K)
                                        N,            // ldb: B^T的leading dimension = N（B的行主序列数）
                                        a,            // A (M×K行主序)，在列主序中是A^T (K×M)
                                        K,            // lda: A^T的leading dimension = K（A的行主序列数）
                                        &beta,        // beta
                                        c,            // C (M×N行主序)，在列主序中是C^T (N×M)
                                        N);           // ldc: C^T的leading dimension = N（C的行主序列数）

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        THROW_RUNTIME_ERROR("cuBLAS SGEMM failed: {}", static_cast<int>(status));
    }

    // 注意：cuBLAS调用是异步的，不需要显式同步
    // 结果会在后续的CUDA操作中自动同步
}

template <>
void cublas_matmul<double>(const double *a, const double *b, double *c, int M, int N, int K)
{
    cublasHandle_t handle = get_cublas_handle();

    const double alpha = 1.0;
    const double beta  = 0.0;

    // cuBLAS使用列主序，要计算行主序的 C = A @ B
    // 对于行主序数据，我们把它们当作列主序的转置：
    // - 行主序 A (M×K) 在列主序中会被解释为 A^T (K×M)
    // - 行主序 B (K×N) 在列主序中会被解释为 B^T (N×K)
    // - 要计算 C = A @ B，在列主序中计算 C^T = B^T @ A^T
    // - 使用 CUBLAS_OP_N，交换 M 和 N，以及 A 和 B 的位置
    // - leading dimension 使用行主序的列数（即矩阵的列数）
    cublasStatus_t status = cublasDgemm(handle,
                                        CUBLAS_OP_N,  // transa: B^T 不转置 (N×K)
                                        CUBLAS_OP_N,  // transb: A^T 不转置 (K×M)
                                        N,            // m: C^T的行数 = B^T的行数 = N
                                        M,            // n: C^T的列数 = A^T的列数 = M
                                        K,            // k: 公共维度 = B^T的列数 = A^T的行数 = K
                                        &alpha,       // alpha
                                        b,            // B (K×N行主序，作为B^T N×K列主序)，ldb = N
                                        N,            // ldb: 行主序B的列数
                                        a,            // A (M×K行主序，作为A^T K×M列主序)，lda = K
                                        K,            // lda: 行主序A的列数
                                        &beta,        // beta
                                        c,            // C (M×N行主序，作为C^T N×M列主序)，ldc = N
                                        N);           // ldc: 行主序C的列数

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        THROW_RUNTIME_ERROR("cuBLAS DGEMM failed: {}", static_cast<int>(status));
    }
}

}  // namespace cuda
}  // namespace origin

#endif  // ENABLE_CUBLAS
