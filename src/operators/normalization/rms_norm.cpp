#include "origin/operators/normalization/rms_norm.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

RMSNorm::RMSNorm(float eps) : eps_(eps) {}

std::vector<Tensor> RMSNorm::forward(const std::vector<Tensor> &xs)
{
    // RMSNorm 需要 2 个输入：x, gamma
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("RMSNorm operator requires exactly 2 inputs (x, gamma), but got {}", xs.size());
    }

    auto &x     = xs[0];
    auto &gamma = xs[1];

    // 获取 Mat 引用
    const Mat &x_mat     = mat(x);
    const Mat &gamma_mat = mat(gamma);

    Tensor y;

    // 检查是否需要梯度计算
    bool any_requires_grad = x.requires_grad() || gamma.requires_grad();
    if (any_requires_grad)
    {
        // 需要梯度计算：使用 rms_norm_forward 保存中间结果
        auto result = x_mat.rms_norm_forward(gamma_mat, eps_);

        // 转换结果
        y          = convert_mat_to_tensor(std::move(result.y));
        saved_rms_ = convert_mat_to_tensor(std::move(result.rms));
    }
    else
    {
        // 不需要梯度计算：使用 rms_norm 只返回输出（节省内存）
        auto result = x_mat.rms_norm(gamma_mat, eps_);
        y           = convert_mat_to_tensor(std::move(result));

        // 不需要保存中间结果
        saved_rms_ = Tensor::zeros(x.shape(), dtype(DataType::kFloat32).device(x.device()));
    }

    std::vector<Tensor> outputs;
    outputs.push_back(std::move(y));
    return outputs;
}

std::vector<Tensor> RMSNorm::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("RMSNorm backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy    = gys[0];
    auto &x     = this->inputs_[0];
    auto &gamma = this->inputs_[1];

    // 获取 Mat 引用并调用底层 rms_norm_backward
    const Mat &gy_mat        = mat(gy);
    const Mat &x_mat         = mat(x);
    const Mat &gamma_mat     = mat(gamma);
    const Mat &saved_rms_mat = mat(saved_rms_);

    auto results = x_mat.rms_norm_backward(gy_mat, gamma_mat, saved_rms_mat, eps_);

    // 转换结果
    auto gx     = convert_mat_to_tensor(std::move(results[0]));
    auto dgamma = convert_mat_to_tensor(std::move(results[1]));

    std::vector<Tensor> outputs;
    outputs.push_back(std::move(gx));
    outputs.push_back(std::move(dgamma));
    return outputs;
}

Tensor rms_norm(const Tensor &x, const Tensor &gamma, float eps)
{
    auto op = std::make_shared<RMSNorm>(eps);
    return (*op)({x, gamma})[0];
}

}  // namespace functional
}  // namespace origin
