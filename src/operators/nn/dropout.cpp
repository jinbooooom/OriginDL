#include <random>
#include <vector>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

Dropout::Dropout(float p, bool training) : p_(p), training_(training)
{
    if (unlikely(p < 0.0f || p >= 1.0f))
    {
        THROW_INVALID_ARG("Dropout: p must be in [0, 1), but got {}", p);
    }
}

std::vector<Tensor> Dropout::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Dropout operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x = xs[0];

    // 获取 Mat 引用
    const Mat &x_mat = mat(x);

    // 根据 requires_grad 决定是否保存 mask
    std::unique_ptr<Mat> result;
    if (x.requires_grad() && training_)
    {
        // 需要梯度计算且训练模式：创建并保存 mask 用于反向传播
        auto mask_mat_unique = x_mat.clone();
        Mat *mask_mat        = mask_mat_unique.get();
        result               = x_mat.dropout(p_, training_, mask_mat);
        mask_                = convert_mat_to_tensor(std::move(mask_mat_unique));
    }
    else
    {
        // 不需要梯度计算或推理模式：不保存 mask，节省内存
        // 在推理模式下，mask_mat 可以为 nullptr
        result = x_mat.dropout(p_, training_, nullptr);
    }

    auto y = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Dropout::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Dropout backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];

    if (!training_)
    {
        // 推理模式：梯度直接传递
        return std::vector<Tensor>{std::move(gy)};
    }

    // 训练模式：根据 mask 计算梯度
    // 如果 mask_ 未初始化（elements() == 0），说明 forward 时没有保存 mask
    // 这种情况不应该发生（因为 requires_grad=true 时应该保存 mask）
    if (unlikely(mask_.elements() == 0))
    {
        THROW_RUNTIME_ERROR(
            "Dropout backward: mask_ is not initialized. This should not happen when requires_grad=true");
    }

    // 获取 Mat 引用
    const Mat &gy_mat   = mat(gy);
    const Mat &mask_mat = mat(mask_);

    // 使用 Mat 接口的 dropout_backward 方法
    // dropout_backward 签名是 dropout_backward(const Mat &gy, const Mat &mask)
    // 所以需要传入 gy_mat 和 mask_mat
    std::unique_ptr<Mat> result = gy_mat.dropout_backward(gy_mat, mask_mat);

    auto gx = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(gx)};
}

}  // namespace functional
}  // namespace origin
