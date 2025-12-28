#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"
#include "origin/utils/conv_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

namespace origin
{
namespace cuda
{

// ==================== CUDA Kernels ====================

/**
 * @brief 计算窗口内的最大值和索引的CUDA kernel
 * @details 对于每个输出位置，在窗口内求最大值并保存索引
 */
template <typename T>
__global__ void max_pool2d_forward_kernel(const T *__restrict__ col_data, T *__restrict__ result_data,
                                           size_t *__restrict__ indices, size_t N, size_t C, size_t OH, size_t OW,
                                           int KH, int KW)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * OH * OW;

    if (idx < total_elements)
    {
        // 计算 (n, c, oh, ow)
        size_t n = idx / (C * OH * OW);
        size_t rem = idx % (C * OH * OW);
        size_t c = rem / (OH * OW);
        rem = rem % (OH * OW);
        size_t oh = rem / OW;
        size_t ow = rem % OW;

        // 在窗口内求最大值和索引
        // col形状: (N, C, KH, KW, OH, OW)
        size_t col_base = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW + oh * OW + ow;
        
        T max_val = col_data[col_base];  // (kh=0, kw=0)
        size_t max_idx = 0;

        for (int kh = 0; kh < KH; ++kh)
        {
            for (int kw = 0; kw < KW; ++kw)
            {
                size_t col_idx = col_base + static_cast<size_t>(kh) * KW * OH * OW +
                                static_cast<size_t>(kw) * OH * OW;
                T val = col_data[col_idx];
                size_t linear_idx = static_cast<size_t>(kh) * KW + static_cast<size_t>(kw);
                if (val > max_val)
                {
                    max_val = val;
                    max_idx = linear_idx;
                }
            }
        }

        result_data[idx] = max_val;
        indices[idx] = max_idx;
    }
}

/**
 * @brief 根据索引分配梯度的CUDA kernel
 * @details 将梯度根据前向传播保存的索引分配到对应位置
 */
template <typename T>
__global__ void max_pool2d_backward_kernel(const T *__restrict__ gy_data, const size_t *__restrict__ indices,
                                           T *__restrict__ gcol_data, size_t N, size_t C, size_t OH, size_t OW, int KH,
                                           int KW)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * OH * OW;

    if (idx < total_elements)
    {
        // 计算 (n, c, oh, ow)
        size_t n = idx / (C * OH * OW);
        size_t rem = idx % (C * OH * OW);
        size_t c = rem / (OH * OW);
        rem = rem % (OH * OW);
        size_t oh = rem / OW;
        size_t ow = rem % OW;

        // 获取索引
        size_t linear_idx = indices[idx];
        int kh = static_cast<int>(linear_idx) / KW;
        int kw = static_cast<int>(linear_idx) % KW;

        // 计算gcol中的位置
        // gcol形状: (N, C, KH, KW, OH, OW)
        size_t gcol_idx = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW +
                         static_cast<size_t>(kh) * KW * OH * OW + static_cast<size_t>(kw) * OH * OW + oh * OW + ow;

        gcol_data[gcol_idx] = gy_data[idx];
    }
}

// ==================== max_pool2d 实现 ====================

std::unique_ptr<Mat> max_pool2d(const OriginMat &x, std::pair<int, int> kernel_size, std::pair<int, int> stride,
                                std::pair<int, int> pad, std::vector<size_t> &indices)
{
    // 输入验证：确保输入是4D张量 (N, C, H, W)
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("max_pool2d: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }

    size_t N = x.shape()[0];
    size_t C = x.shape()[1];
    size_t H = x.shape()[2];
    size_t W = x.shape()[3];

    int KH = kernel_size.first;
    int KW = kernel_size.second;
    int SH = stride.first;
    int SW = stride.second;
    int PH = pad.first;
    int PW = pad.second;

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), KH, SH, PH);
    int OW = get_conv_outsize(static_cast<int>(W), KW, SW, PW);

    if (OH <= 0 || OW <= 0)
    {
        THROW_INVALID_ARG("max_pool2d: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
                         "stride=({},{}), pad=({},{})",
                         OH, OW, H, W, KH, KW, SH, SW, PH, PW);
    }

    // 1. 使用 im2col 提取窗口（to_matrix=false），得到 (N, C, KH, KW, OH, OW)
    auto col = im2col(x, kernel_size, stride, pad, false);
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);

    // 2. 在 (KH, KW) 维度上求最大值，并保存索引
    Shape output_shape{N, C, static_cast<size_t>(OH), static_cast<size_t>(OW)};
    auto result = std::make_unique<OriginMat>(output_shape, x.dtype(), x.device());

    // 清空并准备索引向量
    indices.clear();
    indices.resize(N * C * OH * OW);

    // 在GPU上分配索引内存
    size_t indices_size = N * C * OH * OW * sizeof(size_t);
    size_t *d_indices = nullptr;
    cudaError_t err = cudaMalloc(&d_indices, indices_size);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA memory allocation failed for indices: {}", cudaGetErrorString(err));
    }

    // 使用类型分发器计算最大值和索引
    if (x.dtype() == DataType::kFloat32)
    {
        const float *col_data = col_mat.data_ptr<float>();
        float *result_data = result->data_ptr<float>();

        size_t total_elements = N * C * OH * OW;
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        max_pool2d_forward_kernel<float><<<num_blocks, threads_per_block>>>(col_data, result_data, d_indices, N, C, OH,
                                                                            OW, KH, KW);
    }
    else
    {
        cudaFree(d_indices);
        THROW_INVALID_ARG("max_pool2d: only float32 is supported, got {}", dtype_to_string(x.dtype()));
    }

    CUDA_CHECK_ASYNC();

    // 将索引从GPU拷贝到CPU
    err = cudaMemcpy(indices.data(), d_indices, indices_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(d_indices);
        THROW_RUNTIME_ERROR("CUDA memory copy failed for indices: {}", cudaGetErrorString(err));
    }

    cudaFree(d_indices);

    return result;
}

std::unique_ptr<Mat> max_pool2d_backward(const OriginMat &gy, const OriginMat &x, std::pair<int, int> kernel_size,
                                         std::pair<int, int> stride, std::pair<int, int> pad,
                                         const std::vector<size_t> &indices)
{
    // 输入验证：确保 gy 形状为 (N, C, OH, OW)
    if (gy.shape().size() != 4)
    {
        THROW_INVALID_ARG("max_pool2d_backward: gy must be 4D (N, C, OH, OW), but got shape {}", gy.shape().to_string());
    }

    size_t N = gy.shape()[0];
    size_t C = gy.shape()[1];
    size_t OH = gy.shape()[2];
    size_t OW = gy.shape()[3];

    int KH = kernel_size.first;
    int KW = kernel_size.second;

    // 验证索引大小
    if (indices.size() != N * C * OH * OW)
    {
        THROW_INVALID_ARG("max_pool2d_backward: indices size mismatch. Expected {}, got {}", N * C * OH * OW,
                         indices.size());
    }

    // 1. 创建零张量 gcol，形状 (N, C, KH, KW, OH, OW) 以匹配 col2im 的期望格式
    Shape gcol_shape{N, C, static_cast<size_t>(KH), static_cast<size_t>(KW), OH, OW};
    auto gcol = std::make_unique<OriginMat>(gcol_shape, gy.dtype(), gy.device());

    // 在GPU上分配索引内存
    size_t indices_size = indices.size() * sizeof(size_t);
    size_t *d_indices = nullptr;
    cudaError_t err = cudaMalloc(&d_indices, indices_size);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA memory allocation failed for indices: {}", cudaGetErrorString(err));
    }

    // 将索引从CPU拷贝到GPU
    err = cudaMemcpy(d_indices, indices.data(), indices_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(d_indices);
        THROW_RUNTIME_ERROR("CUDA memory copy failed for indices: {}", cudaGetErrorString(err));
    }

    // 2. 根据索引将 gy 的值放到对应位置
    if (gy.dtype() == DataType::kFloat32)
    {
        const float *gy_data = gy.data_ptr<float>();
        float *gcol_data = gcol->data_ptr<float>();

        // 初始化gcol为0
        cudaMemset(gcol_data, 0, N * C * OH * OW * KH * KW * sizeof(float));

        size_t total_elements = N * C * OH * OW;
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        max_pool2d_backward_kernel<float><<<num_blocks, threads_per_block>>>(gy_data, d_indices, gcol_data, N, C, OH,
                                                                             OW, KH, KW);
    }
    else
    {
        cudaFree(d_indices);
        THROW_INVALID_ARG("max_pool2d_backward: only float32 is supported, got {}", dtype_to_string(gy.dtype()));
    }

    CUDA_CHECK_ASYNC();

    cudaFree(d_indices);

    const OriginMat &gcol_mat = static_cast<const OriginMat &>(*gcol);

    // 3. 使用 col2im 转换回 (N, C, H, W)
    auto gx = col2im(gcol_mat, x.shape(), kernel_size, stride, pad, false);
    return gx;
}

}  // namespace cuda
}  // namespace origin

