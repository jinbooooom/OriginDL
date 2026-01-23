#include <cmath>
#include <vector>
#include "origin/core/config.h"
#include "origin/core/operator.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

BatchNorm::BatchNorm(bool training, float eps, float momentum, int num_dims)
    : training_(training), eps_(eps), momentum_(momentum), num_dims_(num_dims)
{}

std::vector<Tensor> BatchNorm::forward(const std::vector<Tensor> &xs)
{
    // BatchNorm 需要 5 个输入：x, gamma, beta, running_mean, running_var
    if (unlikely(xs.size() != 5))
    {
        THROW_RUNTIME_ERROR(
            "BatchNorm operator requires exactly 5 inputs (x, gamma, beta, running_mean, running_var), but got {}",
            xs.size());
    }

    auto &x            = xs[0];
    auto &gamma        = xs[1];
    auto &beta         = xs[2];
    auto &running_mean = xs[3];
    auto &running_var  = xs[4];

    // 获取 Mat 引用
    const Mat &x_mat            = mat(x);
    const Mat &gamma_mat        = mat(gamma);
    const Mat &beta_mat         = mat(beta);
    const Mat &running_mean_mat = mat(running_mean);
    const Mat &running_var_mat  = mat(running_var);

    Tensor y;

    // 检查是否需要梯度计算：使用 tensor.requires_grad() 方法判断
    // 如果输入 tensor 需要梯度，使用 batch_norm_forward 保存中间结果
    // 如果输入 tensor 不需要梯度，使用 batch_norm 只返回输出（节省内存）
    if (x.requires_grad())
    {
        // 需要梯度计算：使用 batch_norm_forward 保存中间结果
        auto result = x_mat.batch_norm_forward(gamma_mat, beta_mat, running_mean_mat, running_var_mat, training_, eps_,
                                               num_dims_);

        // 转换结果
        y = convert_mat_to_tensor(std::move(result.y));

        // 保存中间结果用于反向传播
        saved_mean_   = convert_mat_to_tensor(std::move(result.mean));
        saved_var_    = convert_mat_to_tensor(std::move(result.var));
        saved_x_norm_ = convert_mat_to_tensor(std::move(result.x_norm));
    }
    else
    {
        // 不需要梯度计算：使用 batch_norm 只返回输出（节省内存）
        auto result = x_mat.batch_norm(gamma_mat, beta_mat, running_mean_mat, running_var_mat, training_, eps_,
                                       momentum_, num_dims_);
        y           = convert_mat_to_tensor(std::move(result));

        // 不需要保存中间结果（反向传播已禁用或输入不在计算图中）
        // 为了安全起见，仍然初始化这些成员（虽然不会被使用）
        saved_mean_   = Tensor::zeros(running_mean.shape(), dtype(DataType::kFloat32).device(x.device()));
        saved_var_    = Tensor::zeros(running_var.shape(), dtype(DataType::kFloat32).device(x.device()));
        saved_x_norm_ = Tensor::zeros(x.shape(), dtype(DataType::kFloat32).device(x.device()));
    }

    std::vector<Tensor> outputs;
    outputs.reserve(training_ ? 3 : 1);
    outputs.push_back(std::move(y));
    // 如果训练模式，返回当前 batch 的 mean 和 var，用于更新 running_mean 和 running_var
    if (training_)
    {
        outputs.push_back(std::move(saved_mean_));  // 当前 batch 的 mean
        outputs.push_back(std::move(saved_var_));   // 当前 batch 的 var
    }
    return outputs;
}

std::vector<Tensor> BatchNorm::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("BatchNorm backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy    = gys[0];
    auto &x     = this->inputs_[0];
    auto &gamma = this->inputs_[1];

    // 获取 Mat 引用并调用底层 batch_norm_backward
    const Mat &gy_mat           = mat(gy);
    const Mat &x_mat            = mat(x);
    const Mat &gamma_mat        = mat(gamma);
    const Mat &saved_mean_mat   = mat(saved_mean_);
    const Mat &saved_var_mat    = mat(saved_var_);
    const Mat &saved_x_norm_mat = mat(saved_x_norm_);

    auto results =
        x_mat.batch_norm_backward(gy_mat, gamma_mat, saved_mean_mat, saved_var_mat, saved_x_norm_mat, eps_, num_dims_);

    // 转换结果
    auto gx     = convert_mat_to_tensor(std::move(results[0]));
    auto dgamma = convert_mat_to_tensor(std::move(results[1]));
    auto dbeta  = convert_mat_to_tensor(std::move(results[2]));

    std::vector<Tensor> outputs;
    outputs.reserve(5);
    outputs.push_back(std::move(gx));
    outputs.push_back(std::move(dgamma));
    outputs.push_back(std::move(dbeta));
    // running_mean 和 running_var 不需要梯度
    outputs.emplace_back(Tensor::zeros(this->inputs_[3].shape(), dtype(DataType::kFloat32).device(x.device())));
    outputs.emplace_back(Tensor::zeros(this->inputs_[4].shape(), dtype(DataType::kFloat32).device(x.device())));
    return outputs;
}

Tensor batch_norm(const Tensor &x,
                  const Tensor &gamma,
                  const Tensor &beta,
                  const Tensor &running_mean,
                  const Tensor &running_var,
                  bool training,
                  float eps,
                  float momentum,
                  int num_dims)
{
    auto op = std::make_shared<BatchNorm>(training, eps, momentum, num_dims);
    return (*op)({x, gamma, beta, running_mean, running_var})[0];
}
}  // namespace functional
}  // namespace origin
