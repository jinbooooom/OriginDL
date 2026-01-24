#include <memory>
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
 * @brief CPU gather：根据索引从矩阵中提取值
 * @param input 输入矩阵 (N, C)
 * @param indices 索引向量 (N,)，每个元素在 [0, C) 范围内
 * @return 提取的值 (N,)
 */
std::unique_ptr<Mat> gather(const OriginMat &input, const OriginMat &indices)
{
    if (unlikely(input.shape().size() != 2))
    {
        THROW_INVALID_ARG("gather: input must be 2D (N, C), but got shape {}", input.shape().to_string());
    }

    if (unlikely(indices.shape().size() != 1))
    {
        THROW_INVALID_ARG("gather: indices must be 1D (N,), but got shape {}", indices.shape().to_string());
    }

    if (unlikely(input.shape()[0] != indices.shape()[0]))
    {
        THROW_INVALID_ARG("gather: batch size mismatch. input has {} samples, indices has {} samples", input.shape()[0],
                          indices.shape()[0]);
    }

    VALIDATE_SAME_CPU_DEVICE(input, indices);

    size_t N = input.shape()[0];
    size_t C = input.shape()[1];

    // 创建输出矩阵
    Shape output_shape{N};
    auto result = std::make_unique<OriginMat>(output_shape, input.dtype(), input.device());

    const void *input_data   = input.storage()->data();
    const void *indices_data = indices.storage()->data();
    void *output_data        = result->storage()->data();

    // 使用类型分发器执行 gather 操作
    device_common::TypeDispatcher::dispatch_void(input.dtype(), [&]<typename T>() {
        const T *input_ptr = static_cast<const T *>(input_data);
        T *output_ptr      = static_cast<T *>(output_data);

        // 根据索引类型提取值
        if (indices.dtype() == DataType::kInt32)
        {
            const int32_t *indices_ptr = static_cast<const int32_t *>(indices_data);
            for (size_t i = 0; i < N; ++i)
            {
                int32_t idx = indices_ptr[i];
                if (unlikely(idx < 0 || idx >= static_cast<int32_t>(C)))
                {
                    THROW_INVALID_ARG("gather: index {} out of range [0, {})", idx, C);
                }
                output_ptr[i] = input_ptr[i * C + idx];
            }
        }
        else if (indices.dtype() == DataType::kInt64)
        {
            const int64_t *indices_ptr = static_cast<const int64_t *>(indices_data);
            for (size_t i = 0; i < N; ++i)
            {
                int64_t idx = indices_ptr[i];
                if (unlikely(idx < 0 || idx >= static_cast<int64_t>(C)))
                {
                    THROW_INVALID_ARG("gather: index {} out of range [0, {})", idx, C);
                }
                output_ptr[i] = input_ptr[i * C + idx];
            }
        }
        else
        {
            THROW_INVALID_ARG("gather: indices must be int32 or int64, but got {}", dtype_to_string(indices.dtype()));
        }
    });

    return result;
}

/**
 * @brief CPU one_hot：将索引转换为 one-hot 编码
 * @param indices 索引向量 (N,)，每个元素在 [0, num_classes) 范围内
 * @param num_classes 类别数量
 * @return one-hot 编码矩阵 (N, num_classes)
 */
std::unique_ptr<Mat> one_hot(const OriginMat &indices, int num_classes)
{
    if (unlikely(indices.shape().size() != 1))
    {
        THROW_INVALID_ARG("one_hot: indices must be 1D (N,), but got shape {}", indices.shape().to_string());
    }

    if (unlikely(num_classes <= 0))
    {
        THROW_INVALID_ARG("one_hot: num_classes must be positive, but got {}", num_classes);
    }

    size_t N = indices.shape()[0];

    // 创建输出矩阵
    Shape output_shape{N, static_cast<size_t>(num_classes)};
    auto result = std::make_unique<OriginMat>(output_shape, DataType::kFloat32, indices.device());

    const void *indices_data = indices.storage()->data();
    void *output_data        = result->storage()->data();
    float *output_ptr        = static_cast<float *>(output_data);

    // 初始化为0
    for (size_t i = 0; i < N * num_classes; ++i)
    {
        output_ptr[i] = 0.0f;
    }

    // 根据索引类型设置 one-hot 编码
    if (indices.dtype() == DataType::kInt32)
    {
        const int32_t *indices_ptr = static_cast<const int32_t *>(indices_data);
        for (size_t i = 0; i < N; ++i)
        {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_classes)
            {
                output_ptr[i * num_classes + idx] = 1.0f;
            }
        }
    }
    else if (indices.dtype() == DataType::kInt64)
    {
        const int64_t *indices_ptr = static_cast<const int64_t *>(indices_data);
        for (size_t i = 0; i < N; ++i)
        {
            int64_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_classes)
            {
                output_ptr[i * num_classes + idx] = 1.0f;
            }
        }
    }
    else
    {
        THROW_INVALID_ARG("one_hot: indices must be int32 or int64, but got {}", dtype_to_string(indices.dtype()));
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
