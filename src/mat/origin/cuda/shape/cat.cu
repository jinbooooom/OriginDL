#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief cat算子实现（在GPU上直接拼接，避免CPU-GPU传输）
 * @param inputs 输入矩阵列表
 * @param dim 拼接维度
 * @return 拼接结果矩阵
 */
std::unique_ptr<Mat> cat(const std::vector<const OriginMat *> &inputs, int dim)
{
    if (unlikely(inputs.empty()))
    {
        THROW_RUNTIME_ERROR("cat: requires at least 1 input");
    }

    if (inputs.size() == 1)
    {
        // 只有一个输入，直接复制
        return std::make_unique<OriginMat>(*inputs[0]);
    }

    // 检查所有输入的形状和设备
    auto first_shape = inputs[0]->shape();
    Device device    = inputs[0]->device();
    DataType dtype   = inputs[0]->dtype();

    for (size_t i = 1; i < inputs.size(); ++i)
    {
        auto shape = inputs[i]->shape();
        if (unlikely(shape.size() != first_shape.size()))
        {
            THROW_RUNTIME_ERROR("cat: all inputs must have same number of dimensions");
        }

        for (size_t d = 0; d < shape.size(); ++d)
        {
            if (unlikely(d != static_cast<size_t>(dim) && shape[d] != first_shape[d]))
            {
                THROW_RUNTIME_ERROR("cat: dimension {} mismatch: {} vs {}", d, shape[d], first_shape[d]);
            }
        }

        VALIDATE_SAME_DTYPE(*inputs[0], *inputs[i]);
        VALIDATE_SAME_CUDA_DEVICE(*inputs[0], *inputs[i]);
    }

    // 计算输出形状
    Shape output_shape    = first_shape;
    size_t total_dim_size = 0;
    for (const auto *x : inputs)
    {
        total_dim_size += x->shape()[dim];
    }
    output_shape[dim] = total_dim_size;

    // 创建输出矩阵
    auto result = std::make_unique<OriginMat>(output_shape, dtype, device);

    size_t element_size_bytes = element_size(dtype);

    // 已经拼接的通道数（在 C 维度上的偏移）
    size_t output_channel_offset = 0;

    for (const auto *input : inputs)
    {
        auto input_shape = input->shape();

        // 思路：将任意维度的 cat 转换为 3 维来思考
        // 例如 [A, B, C, H, W] 在 dim=2 上 cat，可以看作 [A*B, C, H*W] 在 dim=1 上 cat
        // 这样可以将 dim 之前的所有维度合并为 M，dim 之后的所有维度合并为 N
        
        // M: dim 之前所有维度的乘积（外层维度大小）
        //    在 3 维视图 [M, C, N] 中，M 是第一个维度，表示需要复制的"块"的数量
        size_t M = 1;
        for (size_t d = 0; d < static_cast<size_t>(dim); ++d)
        {
            M *= input_shape[d];
        }
        
        // C: cat 维度的大小（通道维度）
        //    在 3 维视图 [M, C, N] 中，C 是第二个维度，是当前输入在拼接维度上的大小
        size_t C = input_shape[dim];
        
        // N: dim 之后所有维度的乘积（内层维度大小）
        //    在 3 维视图 [M, C, N] 中，N 是第三个维度，表示每个通道内的元素数量
        size_t N = 1;
        for (size_t d = dim + 1; d < output_shape.size(); ++d)
        {
            N *= output_shape[d];
        }
        
        // 转换为 3 维后：形状为 [M, C, N]，在 dim=1 (C维度) 上 cat
        // 每个 m_idx 对应 M 维度的一个索引，表示一个连续的内存块（chunk）
        // 每个 chunk 包含 C 个通道，每个通道有 N 个元素，总共 C*N 个元素
        size_t input_chunk_elements  = C * N;  // 输入中一个 chunk 的元素数量
        size_t input_chunk_bytes     = input_chunk_elements * element_size_bytes;
        size_t output_chunk_elements = output_shape[dim] * N;  // 输出中一个 chunk 的元素数量
        
        // 转换为字节指针，避免后续多次转换
        const uint8_t *src = static_cast<const uint8_t *>(input->storage()->data());
        uint8_t *dst_base  = static_cast<uint8_t *>(result->storage()->data());

        // 优化：当 M=1 时（dim=0 的情况），只有一个 chunk，可以一次性拷贝整个输入
        if (M == 1)
        {
            // 输入就是整个矩阵，输出地址 = output_channel_offset * N
            uint8_t *dst_chunk = dst_base + (output_channel_offset * N) * element_size_bytes;
            
            // 一次性拷贝整个输入
            size_t total_input_bytes = input_chunk_bytes;  // M=1 时，只有一个 chunk
            CUDA_CHECK(cudaMemcpyAsync(dst_chunk, src, total_input_bytes, cudaMemcpyDeviceToDevice, nullptr));
        }
        else
        {
            // 通用情况：遍历每个 chunk（对应 M 维度的每个索引）
            for (size_t m_idx = 0; m_idx < M; ++m_idx)
            {
                // 输入地址偏移 = m_idx * C * N（每个 chunk 包含 C*N 个元素）
                const uint8_t *src_chunk = src + m_idx * input_chunk_bytes;
                
                // 输出地址偏移 = m_idx * output_C * N + output_channel_offset * N
                // 其中 output_channel_offset 是已经拼接的通道数（在 C 维度上的偏移）
                uint8_t *dst_chunk =
                    dst_base + (m_idx * output_chunk_elements + output_channel_offset * N) * element_size_bytes;
                
                // 使用异步拷贝，默认流（NULL）允许 GPU 并行执行多个拷贝操作
                CUDA_CHECK(cudaMemcpyAsync(dst_chunk, src_chunk, input_chunk_bytes, cudaMemcpyDeviceToDevice, nullptr));
            }
        }

        // 更新输出偏移（累计已拼接的通道数）
        output_channel_offset += C;
    }

    CUDA_CHECK_ASYNC();
    return result;
}

}  // namespace cuda
}  // namespace origin
