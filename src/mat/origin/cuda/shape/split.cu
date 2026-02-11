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
 * @param split_sizes 沿 dim 维度的各段大小列表
 * @param dim 分割维度
 * @return 分割后的矩阵列表
 */
std::vector<std::unique_ptr<Mat>> split(const OriginMat &input, const std::vector<size_t> &split_sizes, int dim)
{
    if (split_sizes.empty())
    {
        THROW_RUNTIME_ERROR("split: requires at least 1 split size");
    }

    auto input_shape = input.shape();

    // 验证 split_sizes 总和等于输入在 dim 维度上的大小
    size_t total_dim_size = 0;
    for (size_t s : split_sizes)
    {
        total_dim_size += s;
    }
    if (total_dim_size != input_shape[dim])
    {
        THROW_RUNTIME_ERROR("split: total split_sizes {} does not match input dimension size {} at dim {}",
                            total_dim_size, input_shape[dim], dim);
    }

    std::vector<std::unique_ptr<Mat>> results;
    results.reserve(split_sizes.size());

    // ============================================================================
    // 算法说明：将多维张量在指定维度上分割（cat 的反向操作）
    //
    // 思路：将任意维度的 split 转换为 3 维来思考（与 cat 对应）
    // 例如：Shape{A, B, C₁+C₂, H, W} 在 dim=2 上 split，可以看作 Shape{A*B, C₁+C₂, H*W} 在 dim=1 上 split
    // 这样可以将 dim 之前的所有维度合并为 M，dim 之后的所有维度合并为 N
    // ============================================================================

    size_t element_size_bytes = element_size(input.dtype());

    // M: dim 之前所有维度的乘积（外层维度大小）
    size_t M = 1;
    for (size_t d = 0; d < static_cast<size_t>(dim); ++d)
    {
        M *= input_shape[d];
    }

    // C: 输入在 dim 维度上的总大小
    size_t C = input_shape[dim];

    // N: dim 之后所有维度的乘积（内层维度大小）
    size_t N = 1;
    for (size_t d = dim + 1; d < input_shape.size(); ++d)
    {
        N *= input_shape[d];
    }

    size_t input_chunk_elements = C * N;

    const uint8_t *src_base     = static_cast<const uint8_t *>(input.storage()->data());
    size_t input_channel_offset = 0;

    for (size_t Ci : split_sizes)
    {
        // 构造输出形状
        Shape output_shape = input_shape;
        output_shape[dim]  = Ci;

        auto result       = std::make_unique<OriginMat>(output_shape, input.dtype(), input.device());
        void *output_data = result->storage()->data();

        size_t output_chunk_elements = Ci * N;
        size_t output_chunk_bytes    = output_chunk_elements * element_size_bytes;

        uint8_t *dst_base = static_cast<uint8_t *>(output_data);

        for (size_t m_idx = 0; m_idx < M; ++m_idx)
        {
            const uint8_t *src_chunk =
                src_base + (m_idx * input_chunk_elements + input_channel_offset * N) * element_size_bytes;
            uint8_t *dst_chunk = dst_base + m_idx * output_chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(dst_chunk, src_chunk, output_chunk_bytes, cudaMemcpyDeviceToDevice, nullptr));
        }

        results.emplace_back(std::move(result));
        input_channel_offset += Ci;
    }

    return results;
}

}  // namespace cuda
}  // namespace origin
