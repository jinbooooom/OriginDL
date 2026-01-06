#include "origin/mat/origin/cuda/cublas_wrapper.h"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/utils/exception.h"
#include <cublas_v2.h>

namespace origin
{
namespace cuda
{

CublasWrapper::CublasWrapper()
{
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        THROW_RUNTIME_ERROR("Failed to create cuBLAS handle: {}", static_cast<int>(status));
    }

    // 设置 cuBLAS 使用 Tensor Core（如果可用）
    status = cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        // Tensor Core 不可用时回退到默认模式
        cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    }
}

CublasWrapper::~CublasWrapper()
{
    if (handle_ != nullptr)
    {
        cublasDestroy(handle_);
    }
}

CublasWrapper &CublasWrapper::get_instance()
{
    static CublasWrapper instance;
    return instance;
}

void CublasWrapper::sgemm(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc,
                           cudaStream_t stream)
{
    // 如果指定了 stream，设置 cuBLAS 使用该 stream
    if (stream != nullptr)
    {
        cublasSetStream(handle_, stream);
    }

    cublasStatus_t status = cublasSgemm(handle_,
                                        transa, transb,
                                        m, n, k,
                                        alpha,
                                        A, lda,
                                        B, ldb,
                                        beta,
                                        C, ldc);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        THROW_RUNTIME_ERROR("cuBLAS SGEMM failed: {}", static_cast<int>(status));
    }
}

}  // namespace cuda
}  // namespace origin

