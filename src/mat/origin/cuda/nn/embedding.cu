#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"
namespace origin
{
namespace cuda
{
/**
 * @brief CUDA embedding (前向传播)
 * @details embedding_dim 个线程处理一个token
 */
template <typename T>
__global__ void embedding_kernel(const int32_t *__restrict__ indices,
                                 const T *__restrict__ vocab,
                                 T *__restrict__ output,
                                 const size_t num_indices,
                                 const int embedding_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices * embedding_dim)
    {
        size_t index_idx = idx / embedding_dim;
        size_t emb_idx   = idx % embedding_dim;
        int32_t token_id = indices[index_idx];
        const T *src     = vocab + token_id * embedding_dim + emb_idx;
        T *dst           = output + idx;
        *dst             = *src;
    }
}
__global__ void embedding_float4_kernel(const int32_t *indices,
                                        const float *vocab,
                                        float *output,
                                        const size_t num_indices,
                                        const int embedding_dim)
{
    const size_t vec_idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec4_per_token = embedding_dim >> 2;  // 每个token对应的float4数量
    const size_t total_vec4  = num_indices * vec4_per_token;

    if (vec_idx < total_vec4)
    {
        const size_t index_idx  = vec_idx / vec4_per_token;
        const int vec4_emb_idx  = vec_idx % vec4_per_token;
        const int emb_dim_start = vec4_emb_idx << 2;  // 内存中的位置

        const int32_t token_id = indices[index_idx];
        const float4 *src      = reinterpret_cast<const float4 *>(vocab + token_id * embedding_dim + emb_dim_start);
        float4 *dst            = reinterpret_cast<float4 *>(output) + vec_idx;  // 直接偏移
        *dst                   = *src;
    }
}
// TODO 针对half优化？

// Backward kernel: float 版本
__global__ void embedding_backward_kernel_float(const float *__restrict__ grad_output,
                                                const int32_t *__restrict__ indices,
                                                float *__restrict__ grad_weight,
                                                const size_t num_indices,
                                                const int embedding_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices * embedding_dim)
    {
        size_t index_idx = idx / embedding_dim;
        int emb_dim_idx  = idx % embedding_dim;

        int32_t token_id      = indices[index_idx];
        const float *grad_src = grad_output + idx;
        float *grad_dst       = grad_weight + token_id * embedding_dim + emb_dim_idx;
        atomicAdd(grad_dst, *grad_src);
    }
}

// Backward kernel: double 版本（使用原子操作 CAS 实现）
__global__ void embedding_backward_kernel_double(const double *__restrict__ grad_output,
                                                 const int32_t *__restrict__ indices,
                                                 double *__restrict__ grad_weight,
                                                 const size_t num_indices,
                                                 const int embedding_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices * embedding_dim)
    {
        size_t index_idx = idx / embedding_dim;
        int emb_dim_idx  = idx % embedding_dim;

        int32_t token_id       = indices[index_idx];
        const double *grad_src = grad_output + idx;
        double *grad_dst       = grad_weight + token_id * embedding_dim + emb_dim_idx;

// 对于 double 类型，使用 atomicAdd（CUDA 11.0+ 支持直接调用）
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 75)
        atomicAdd(grad_dst, *grad_src);
#else
        // 如果硬件不支持，使用简单的原子操作循环
        unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(grad_dst);
        unsigned long long int old             = *address_as_ull;
        unsigned long long int assumed;
        do
        {
            assumed                        = old;
            unsigned long long int new_val = __double_as_longlong(__longlong_as_double(assumed) + *grad_src);
            old                            = atomicCAS(address_as_ull, assumed, new_val);
        } while (assumed != old);
#endif
    }
}

// === 启动函数 ===
/**
 * @brief CUDA embedding：Emebdding 前向传播
 * @param x 输入张量
 * @param vocab 词表
 * @return out 输出张量
 */
std::unique_ptr<Mat> embedding(const OriginMat &x, const OriginMat &vocab)
{
    if (unlikely(x.elements() == 0 || vocab.elements() == 0))
    {
        THROW_INVALID_ARG("X or vocab is empty");
    }
    VALIDATE_CUDA_DEVICE(x);
    VALIDATE_CUDA_DEVICE(vocab);
    const auto &indices_shape = x.shape();
    const int embedding_dim   = static_cast<int>(vocab.shape()[1]);

    std::vector<size_t> out_shape = indices_shape.dims();
    out_shape.push_back(embedding_dim);
    Shape output_shape(out_shape);
    auto result_unique = std::make_unique<OriginMat>(output_shape, vocab.dtype(), vocab.device());

    const void *vocab_data  = vocab.storage()->data();
    const int *indices_data = static_cast<const int32_t *>(x.storage()->data());
    void *output_data       = result_unique->storage()->data();

    const size_t num_indices = indices_shape.elements();
    // 使用float4版本 针对float32类型
    if (vocab.dtype() == DataType::kFloat32 && embedding_dim % 4 == 0)
    {
        constexpr size_t threads_per_block = 256;
        const int vec4_per_token           = embedding_dim >> 2;
        const size_t total_vec4            = num_indices * vec4_per_token;
        const size_t num_blocks            = (total_vec4 + threads_per_block - 1) / threads_per_block;

        embedding_float4_kernel<<<num_blocks, threads_per_block>>>(
            static_cast<const int32_t *>(indices_data), static_cast<const float *>(vocab_data),
            static_cast<float *>(output_data), num_indices, embedding_dim);
    }
    else
    {
        constexpr size_t threads_per_block = 256;
        const size_t total_elements        = num_indices * embedding_dim;
        const size_t num_blocks            = (total_elements + threads_per_block - 1) / threads_per_block;

        device_common::TypeDispatcher::dispatch_void(vocab.dtype(), [&]<typename T>() {
            embedding_kernel<T><<<num_blocks, threads_per_block>>>(
                static_cast<const int32_t *>(indices_data), static_cast<const T *>(vocab_data),
                static_cast<T *>(output_data), num_indices, embedding_dim);
        });
    }
    CUDA_CHECK_ASYNC();
    return result_unique;
}

/**
 * @brief CUDA embedding_backward: Embedding 反向传播
 * @param gy 输出梯度
 * @param x 输入张量
 * @param vocab_size 词表大小
 * @param embedding_dim 嵌入维度
 * @return 权重梯度(vocab_size, embedding_dim)
 */
std::unique_ptr<Mat> embedding_backward(const OriginMat &gy, const OriginMat &x, int vocab_size, int embedding_dim)
{
    VALIDATE_CUDA_DEVICE(gy);
    VALIDATE_CUDA_DEVICE(x);

    // 权重梯度创建
    Shape grad_weight_shape{static_cast<size_t>(vocab_size), static_cast<size_t>(embedding_dim)};
    auto grad_weight_unique = std::make_unique<OriginMat>(grad_weight_shape, gy.dtype(), gy.device());

    // 获取数据指针
    const void *grad_output_data = gy.storage()->data();
    const void *indices_data     = x.storage()->data();
    void *grad_weight_data       = grad_weight_unique->storage()->data();

    const size_t num_indices    = x.elements();
    const size_t total_elements = num_indices * embedding_dim;

    // 初始化梯度为零
    const size_t grad_weight_bytes = vocab_size * embedding_dim * element_size(gy.dtype());
    CUDA_CHECK(cudaMemset(grad_weight_data, 0, grad_weight_bytes));

    // 启动反向传播 kernel
    constexpr size_t threads_per_block = 256;
    const size_t num_blocks            = (total_elements + threads_per_block - 1) / threads_per_block;

    // 根据数据类型调用不同的 kernel
    if (gy.dtype() == DataType::kFloat32)
    {
        embedding_backward_kernel_float<<<num_blocks, threads_per_block>>>(
            static_cast<const float *>(grad_output_data), static_cast<const int32_t *>(indices_data),
            static_cast<float *>(grad_weight_data), num_indices, embedding_dim);
    }
    else if (gy.dtype() == DataType::kFloat64)
    {
        embedding_backward_kernel_double<<<num_blocks, threads_per_block>>>(
            static_cast<const double *>(grad_output_data), static_cast<const int32_t *>(indices_data),
            static_cast<double *>(grad_weight_data), num_indices, embedding_dim);
    }
    else
    {
        THROW_INVALID_ARG("Unsupported dtype {} for embedding backward", dtype_to_string(gy.dtype()));
    }

    CUDA_CHECK_ASYNC();
    return grad_weight_unique;
}

}  // namespace cuda
}  // namespace origin