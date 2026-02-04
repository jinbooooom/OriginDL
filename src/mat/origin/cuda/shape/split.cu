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
 * @brief CUDA split：将矩阵沿指定维度分割成多个矩阵（cat 的反向操作）
 * @param input 输入矩阵
 * @param output_shapes 输出形状列表
 * @param dim 分割维度
 * @return 分割后的矩阵列表
 */
std::vector<std::unique_ptr<Mat>> split(const OriginMat &input, const std::vector<Shape> &output_shapes, int dim)
{
    if (output_shapes.empty())
    {
        THROW_RUNTIME_ERROR("split: requires at least 1 output shape");
    }

    auto input_shape = input.shape();

    // 验证输出形状的总和等于输入形状
    size_t total_dim_size = 0;
    for (const auto &shape : output_shapes)
    {
        if (shape.size() != input_shape.size())
        {
            THROW_RUNTIME_ERROR("split: output shape dimension mismatch. Expected {}, got {}", input_shape.size(),
                                shape.size());
        }
        for (size_t d = 0; d < shape.size(); ++d)
        {
            if (d != static_cast<size_t>(dim) && shape[d] != input_shape[d])
            {
                THROW_RUNTIME_ERROR("split: dimension {} mismatch: {} vs {}", d, shape[d], input_shape[d]);
            }
        }
        total_dim_size += shape[dim];
    }
    if (total_dim_size != input_shape[dim])
    {
        THROW_RUNTIME_ERROR("split: total output dimension size {} does not match input dimension size {}",
                            total_dim_size, input_shape[dim]);
    }

    std::vector<std::unique_ptr<Mat>> results;
    results.reserve(output_shapes.size());

    // ============================================================================
    // 算法说明：将多维张量在指定维度上分割（cat 的反向操作）
    // 
    // 思路：将任意维度的 split 转换为 3 维来思考（与 cat 对应）
    // 例如：Shape{A, B, C₁+C₂, H, W} 在 dim=2 上 split，可以看作 Shape{A*B, C₁+C₂, H*W} 在 dim=1 上 split
    // 这样可以将 dim 之前的所有维度合并为 M，dim 之后的所有维度合并为 N
    // ============================================================================

    size_t element_size_bytes = element_size(input.dtype());

    // M: dim 之前所有维度的乘积（外层维度大小）
    //    在 3 维视图 [M, C, N] 中，M 是第一个维度，表示需要复制的"块"的数量
    size_t M = 1;
    for (size_t d = 0; d < static_cast<size_t>(dim); ++d)
    {
        M *= input_shape[d];
    }

    // C: 输入在 dim 维度上的总大小
    //    在 3 维视图 [M, C, N] 中，C 是第二个维度，是输入在分割维度上的总大小
    size_t C = input_shape[dim];

    // N: dim 之后所有维度的乘积（内层维度大小）
    //    在 3 维视图 [M, C, N] 中，N 是第三个维度，表示每个通道内的元素数量
    size_t N = 1;
    for (size_t d = dim + 1; d < input_shape.size(); ++d)
    {
        N *= input_shape[d];
    }

    // 转换为 3 维后：形状为 [M, C, N]，在 dim=1 (C维度) 上 split
    // 每个 m_idx 对应 M 维度的一个索引，表示一个连续的内存块（chunk）
    // 每个 chunk 包含 C 个通道，每个通道有 N 个元素，总共 C*N 个元素
    size_t input_chunk_elements = C * N;  // 输入中一个 chunk 的元素数量

    const uint8_t *src_base = static_cast<const uint8_t *>(input.storage()->data());
    
    // input_channel_offset: 在 C 维度上的偏移量（以通道为单位）
    // 用于跟踪当前输出应该从输入的 C 维度的哪个位置开始复制
    size_t input_channel_offset = 0;

    // 遍历每个输出形状
    for (const auto &output_shape : output_shapes)
    {
        // 创建输出矩阵
        auto result = std::make_unique<OriginMat>(output_shape, input.dtype(), input.device());
        void *output_data = result->storage()->data();

        // Ci: 第 i 个输出在 dim 维度上的大小（通道数）
        //    在 3 维视图 [M, Ci, N] 中，Ci 是当前输出在分割维度上的大小
        size_t Ci = output_shape[dim];

        // output_chunk_elements: 输出中一个 chunk 的元素数量
        size_t output_chunk_elements = Ci * N;
        size_t output_chunk_bytes    = output_chunk_elements * element_size_bytes;

        uint8_t *dst_base = static_cast<uint8_t *>(output_data);

        for (size_t m_idx = 0; m_idx < M; ++m_idx)
        {
            // 输入地址偏移的元素数量 = m_idx * input_chunk_elements + input_channel_offset * N
            // 其中 input_channel_offset 是已经处理的通道数（在 C 维度上的偏移）
            const uint8_t *src_chunk =
                src_base + (m_idx * input_chunk_elements + input_channel_offset * N) * element_size_bytes;
            
            uint8_t *dst_chunk = dst_base + m_idx * output_chunk_bytes;
            
            CUDA_CHECK(cudaMemcpyAsync(dst_chunk, src_chunk, output_chunk_bytes, cudaMemcpyDeviceToDevice, nullptr));
        }

        results.emplace_back(std::move(result));
        // 更新偏移量，为下一个输出做准备
        // 例如：第1个输出复制了 Ci 个通道，则 input_channel_offset 增加 Ci
        input_channel_offset += Ci;
    }

    return results;
}

}  // namespace cuda
}  // namespace origin
