#include <cstring>
#include <memory>
#include <random>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU embedding: Embedding 前向传播
 * @param x 输入张量（索引，必须是 int32 类型）
 * @param vocab 词表（权重矩阵，支持任意数据类型）
 * @return 输出张量（数据类型与 vocab 相同）
 */
std::unique_ptr<Mat> embedding(const OriginMat &x, const OriginMat &vocab)
{
    if (unlikely(x.elements() == 0 || vocab.elements() == 0))
        THROW_INVALID_ARG("Input or Vocab is empty!");
    VALIDATE_CPU_DEVICE(x);
    VALIDATE_CPU_DEVICE(vocab);

    // 验证索引的数据类型必须是 int32
    if (unlikely(x.dtype() != DataType::kInt32))
    {
        THROW_INVALID_ARG("Embedding input indices must be int32, but got {}", static_cast<int>(x.dtype()));
    }

    const auto &indices_shape = x.shape();
    const int vocab_size      = static_cast<int>(vocab.shape()[0]);
    const int embedding_dim   = static_cast<int>(vocab.shape()[1]);

    std::vector<size_t> out_shape = indices_shape.dims();
    out_shape.push_back(embedding_dim);
    Shape output_shape(out_shape);

    auto result_unique = std::make_unique<OriginMat>(output_shape, vocab.dtype(), vocab.device());

    // 获取数据指针
    const void *vocab_data   = vocab.storage()->data();
    const void *indices_data = x.storage()->data();
    void *output_data        = result_unique->storage()->data();

    const size_t num_indices = indices_shape.elements();

    // 使用类型分发器支持多种数据类型
    device_common::TypeDispatcher::dispatch_void(vocab.dtype(), [&]<typename T>() {
        const T *vocab_ptr         = static_cast<const T *>(vocab_data);
        const int32_t *indices_ptr = static_cast<const int32_t *>(indices_data);
        T *output_ptr              = static_cast<T *>(output_data);

        // 逐个复制 embedding 向量
        for (size_t i = 0; i < num_indices; ++i)
        {
            int32_t token_id = indices_ptr[i];
            // 边界检查
            if (unlikely(token_id < 0 || token_id >= vocab_size))
            {
                THROW_INVALID_ARG("Token ID out of range: {} (vocab_size={})", token_id, vocab_size);
            }
            const T *src = vocab_ptr + token_id * embedding_dim;
            T *dst       = output_ptr + i * embedding_dim;
            std::memcpy(dst, src, embedding_dim * sizeof(T));
        }
    });

    return result_unique;
}
/**
 * @brief CPU embedding_backward: Embedding 反向传播
 * @param grad_output 输出梯度（支持任意数据类型）
 * @param x 输入索引张量（必须是 int32 类型）
 * @param vocab_size 词表大小
 * @param embedding_dim 嵌入维度
 * @return 权重梯度(vocab_size, embedding_dim)，数据类型与 grad_output 相同
 */
std::unique_ptr<Mat> embedding_backward(const OriginMat &grad_output,
                                        const OriginMat &x,
                                        int vocab_size,
                                        int embedding_dim)
{
    VALIDATE_CPU_DEVICE(grad_output);
    VALIDATE_CPU_DEVICE(x);

    // 验证索引的数据类型必须是 int32
    if (unlikely(x.dtype() != DataType::kInt32))
    {
        THROW_INVALID_ARG("Embedding input indices must be int32, but got {}", static_cast<int>(x.dtype()));
    }

    // 输出形状
    const auto &output_shape = grad_output.shape();

    // 创建权重梯度
    Shape grad_weight_shape{static_cast<size_t>(vocab_size), static_cast<size_t>(embedding_dim)};
    auto grad_weight_unique = std::make_unique<OriginMat>(grad_weight_shape, grad_output.dtype(), grad_output.device());

    // 获取数据指针
    const void *grad_output_data = grad_output.storage()->data();
    const void *indices_data     = x.storage()->data();
    void *grad_weight_data       = grad_weight_unique->storage()->data();

    const size_t num_tokens = x.elements();

    // 使用类型分发器支持多种数据类型
    device_common::TypeDispatcher::dispatch_void(grad_output.dtype(), [&]<typename T>() {
        const T *grad_output_ptr   = static_cast<const T *>(grad_output_data);
        const int32_t *indices_ptr = static_cast<const int32_t *>(indices_data);
        T *grad_weight_ptr         = static_cast<T *>(grad_weight_data);

        // 初始化为零
        std::fill(grad_weight_ptr, grad_weight_ptr + vocab_size * embedding_dim, T(0));

        // Scatter Add: 累加梯度到对应的行
        for (size_t i = 0; i < num_tokens; ++i)
        {
            int32_t token_id = indices_ptr[i];
            if (unlikely(token_id < 0 || token_id >= vocab_size))
            {
                THROW_INVALID_ARG("Token ID out of range: {} (vocab_size={})", token_id, vocab_size);
            }
            const T *grad_src = grad_output_ptr + i * embedding_dim;
            T *grad_dst       = grad_weight_ptr + token_id * embedding_dim;

            // 累加梯度
            for (int j = 0; j < embedding_dim; ++j)
            {
                grad_dst[j] += grad_src[j];
            }
        }
    });

    return grad_weight_unique;
}

}  // namespace cpu
}  // namespace origin