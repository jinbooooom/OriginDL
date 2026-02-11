#include "origin/utils/metrics.h"
#include <algorithm>
#include <cmath>
#include "origin/core/operator.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{

Tensor accuracy(const Tensor &y, const Tensor &target)
{
    auto y_shape      = y.shape();
    auto target_shape = target.shape();

    // 验证输入形状
    if (unlikely(y_shape.size() != 2))
    {
        THROW_INVALID_ARG("accuracy expects y to be 2D (N, C), but got shape {}", y_shape.to_string());
    }
    if (unlikely(target_shape.size() != 1))
    {
        THROW_INVALID_ARG("accuracy expects target to be 1D (N,), but got shape {}", target_shape.to_string());
    }
    if (unlikely(y_shape[0] != target_shape[0]))
    {
        THROW_INVALID_ARG("accuracy: batch size mismatch. y has {} samples, target has {} samples", y_shape[0],
                          target_shape[0]);
    }

    size_t N = y_shape[0];  // batch size
    size_t C = y_shape[1];  // number of classes

    // 获取数据
    auto y_data      = y.to_vector<float>();
    auto target_data = target.to_vector<int32_t>();

    // 计算准确率：accuracy = mean(argmax(y, axis=1) == target)
    size_t correct = 0;
    for (size_t i = 0; i < N; ++i)
    {
        // 找到第 i 个样本的预测类别（argmax）
        size_t max_idx = 0;
        float max_val  = y_data[i * C];
        for (size_t j = 1; j < C; ++j)
        {
            float val = y_data[i * C + j];
            if (val > max_val)
            {
                max_val = val;
                max_idx = j;
            }
        }

        // 检查预测是否正确
        int32_t t = target_data[i];
        if (t >= 0 && t < static_cast<int32_t>(C) && static_cast<size_t>(t) == max_idx)
        {
            correct++;
        }
    }

    // 计算准确率
    float accuracy_value = static_cast<float>(correct) / static_cast<float>(N);

    // 创建标量准确率张量
    auto result = Tensor({accuracy_value}, Shape{}, dtype(DataType::kFloat32).device(y.device()));

    return result;
}

}  // namespace origin
