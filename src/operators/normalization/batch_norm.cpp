#include <cmath>
#include <vector>
#include "origin/core/operator.h"
#include "origin/mat/mat.h"
#include "origin/utils/exception.h"

namespace origin
{

BatchNorm::BatchNorm(bool training, float eps, float momentum, int num_dims)
    : training_(training), eps_(eps), momentum_(momentum), num_dims_(num_dims)
{}

std::vector<Tensor> BatchNorm::forward(const std::vector<Tensor> &xs)
{
    // BatchNorm 需要 5 个输入：x, gamma, beta, running_mean, running_var
    if (xs.size() != 5)
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

    auto x_shape           = x.shape();
    auto x_data            = x.to_vector<float>();
    auto gamma_data        = gamma.to_vector<float>();
    auto beta_data         = beta.to_vector<float>();
    auto running_mean_data = running_mean.to_vector<float>();
    auto running_var_data  = running_var.to_vector<float>();

    size_t num_channels = x_shape[1];  // 通道在维度1

    // 计算需要求均值的元素数量（沿 batch 和其他空间维度）
    size_t reduce_size = x_shape[0];  // batch size
    if (num_dims_ == 4)               // BatchNorm2d: (N, C, H, W)
    {
        reduce_size *= x_shape[2] * x_shape[3];  // H * W
    }

    std::vector<float> mean_data(num_channels, 0.0f);
    std::vector<float> var_data(num_channels, 0.0f);

    if (training_)
    {
        // 训练模式：计算当前 batch 的均值和方差
        // 对每个通道计算均值和方差
        for (size_t c = 0; c < num_channels; ++c)
        {
            float sum    = 0.0f;
            size_t count = 0;

            // 遍历所有元素，找到属于通道 c 的元素
            if (num_dims_ == 2)  // BatchNorm1d: (N, C)
            {
                for (size_t n = 0; n < x_shape[0]; ++n)
                {
                    size_t idx = n * num_channels + c;
                    sum += x_data[idx];
                    count++;
                }
            }
            else if (num_dims_ == 4)  // BatchNorm2d: (N, C, H, W)
            {
                for (size_t n = 0; n < x_shape[0]; ++n)
                {
                    for (size_t h = 0; h < x_shape[2]; ++h)
                    {
                        for (size_t w = 0; w < x_shape[3]; ++w)
                        {
                            size_t idx = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                            sum += x_data[idx];
                            count++;
                        }
                    }
                }
            }

            if (count > 0)
            {
                mean_data[c] = sum / static_cast<float>(count);
            }

            // 计算方差
            float sum_sq_diff = 0.0f;
            if (num_dims_ == 2)
            {
                for (size_t n = 0; n < x_shape[0]; ++n)
                {
                    size_t idx = n * num_channels + c;
                    float diff = x_data[idx] - mean_data[c];
                    sum_sq_diff += diff * diff;
                }
            }
            else if (num_dims_ == 4)
            {
                for (size_t n = 0; n < x_shape[0]; ++n)
                {
                    for (size_t h = 0; h < x_shape[2]; ++h)
                    {
                        for (size_t w = 0; w < x_shape[3]; ++w)
                        {
                            size_t idx = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                            float diff = x_data[idx] - mean_data[c];
                            sum_sq_diff += diff * diff;
                        }
                    }
                }
            }

            if (count > 0)
            {
                var_data[c] = sum_sq_diff / static_cast<float>(count);
            }
        }
    }
    else
    {
        // 推理模式：使用 running_mean 和 running_var
        mean_data = running_mean_data;
        var_data  = running_var_data;
    }

    // 归一化：x_norm = (x - mean) / sqrt(var + eps)
    std::vector<float> x_norm_data(x_data.size());

    if (num_dims_ == 2)
    {
        for (size_t n = 0; n < x_shape[0]; ++n)
        {
            for (size_t c = 0; c < num_channels; ++c)
            {
                size_t idx       = n * num_channels + c;
                float mean_val   = mean_data[c];
                float var_val    = var_data[c];
                float std_val    = std::sqrt(var_val + eps_);
                x_norm_data[idx] = (x_data[idx] - mean_val) / std_val;
            }
        }
    }
    else if (num_dims_ == 4)
    {
        for (size_t n = 0; n < x_shape[0]; ++n)
        {
            for (size_t c = 0; c < num_channels; ++c)
            {
                float mean_val = mean_data[c];
                float var_val  = var_data[c];
                float std_val  = std::sqrt(var_val + eps_);
                for (size_t h = 0; h < x_shape[2]; ++h)
                {
                    for (size_t w = 0; w < x_shape[3]; ++w)
                    {
                        size_t idx       = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                        x_norm_data[idx] = (x_data[idx] - mean_val) / std_val;
                    }
                }
            }
        }
    }

    // 应用 gamma 和 beta：y = gamma * x_norm + beta
    std::vector<float> y_data(x_data.size());
    for (size_t i = 0; i < x_data.size(); ++i)
    {
        size_t c = 0;
        if (num_dims_ == 2)
        {
            c = (i % num_channels);
        }
        else if (num_dims_ == 4)
        {
            c = ((i / (x_shape[2] * x_shape[3])) % num_channels);
        }
        y_data[i] = gamma_data[c] * x_norm_data[i] + beta_data[c];
    }

    // 创建输出张量
    auto y = Tensor(y_data, x_shape, dtype(DataType::kFloat32).device(x.device()));

    // 保存中间结果用于反向传播
    saved_mean_   = Tensor(mean_data, running_mean.shape(), dtype(DataType::kFloat32).device(x.device()));
    saved_var_    = Tensor(var_data, running_var.shape(), dtype(DataType::kFloat32).device(x.device()));
    saved_x_norm_ = Tensor(x_norm_data, x_shape, dtype(DataType::kFloat32).device(x.device()));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    // 如果训练模式，返回当前 batch 的 mean 和 var，用于更新 running_mean 和 running_var
    if (training_)
    {
        outputs.push_back(saved_mean_);  // 当前 batch 的 mean
        outputs.push_back(saved_var_);   // 当前 batch 的 var
    }
    return outputs;
}

std::vector<Tensor> BatchNorm::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("BatchNorm backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy    = gys[0];
    auto &x     = this->inputs_[0];
    auto &gamma = this->inputs_[1];

    auto gy_data           = gy.to_vector<float>();
    auto x_shape           = x.shape();
    auto gamma_data        = gamma.to_vector<float>();
    auto saved_mean_data   = saved_mean_.to_vector<float>();
    auto saved_var_data    = saved_var_.to_vector<float>();
    auto saved_x_norm_data = saved_x_norm_.to_vector<float>();

    size_t num_channels = x_shape[1];

    // 计算梯度
    std::vector<float> gx_data(x_shape.elements(), 0.0f);
    std::vector<float> dgamma_data(num_channels, 0.0f);
    std::vector<float> dbeta_data(num_channels, 0.0f);

    // 计算每个通道的统计量
    for (size_t c = 0; c < num_channels; ++c)
    {
        float sum_gy       = 0.0f;
        float sum_gy_xnorm = 0.0f;
        size_t count       = 0;

        if (num_dims_ == 2)
        {
            for (size_t n = 0; n < x_shape[0]; ++n)
            {
                size_t idx = n * num_channels + c;
                sum_gy += gy_data[idx];
                sum_gy_xnorm += gy_data[idx] * saved_x_norm_data[idx];
                count++;
            }
        }
        else if (num_dims_ == 4)
        {
            for (size_t n = 0; n < x_shape[0]; ++n)
            {
                for (size_t h = 0; h < x_shape[2]; ++h)
                {
                    for (size_t w = 0; w < x_shape[3]; ++w)
                    {
                        size_t idx = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                        sum_gy += gy_data[idx];
                        sum_gy_xnorm += gy_data[idx] * saved_x_norm_data[idx];
                        count++;
                    }
                }
            }
        }

        float mean_gy       = (count > 0) ? sum_gy / static_cast<float>(count) : 0.0f;
        float mean_gy_xnorm = (count > 0) ? sum_gy_xnorm / static_cast<float>(count) : 0.0f;
        float std_val       = std::sqrt(saved_var_data[c] + eps_);

        // 计算 dgamma 和 dbeta
        dgamma_data[c] = sum_gy_xnorm;
        dbeta_data[c]  = sum_gy;

        // 计算 gx
        if (num_dims_ == 2)
        {
            for (size_t n = 0; n < x_shape[0]; ++n)
            {
                size_t idx = n * num_channels + c;
                float gx_val =
                    (gamma_data[c] / std_val) * (gy_data[idx] - mean_gy - saved_x_norm_data[idx] * mean_gy_xnorm);
                gx_data[idx] = gx_val;
            }
        }
        else if (num_dims_ == 4)
        {
            for (size_t n = 0; n < x_shape[0]; ++n)
            {
                for (size_t h = 0; h < x_shape[2]; ++h)
                {
                    for (size_t w = 0; w < x_shape[3]; ++w)
                    {
                        size_t idx   = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                        float gx_val = (gamma_data[c] / std_val) *
                                       (gy_data[idx] - mean_gy - saved_x_norm_data[idx] * mean_gy_xnorm);
                        gx_data[idx] = gx_val;
                    }
                }
            }
        }
    }

    auto gx     = Tensor(gx_data, x_shape, dtype(DataType::kFloat32).device(x.device()));
    auto dgamma = Tensor(dgamma_data, gamma.shape(), dtype(DataType::kFloat32).device(x.device()));
    auto dbeta  = Tensor(dbeta_data, gamma.shape(), dtype(DataType::kFloat32).device(x.device()));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    outputs.push_back(dgamma);
    outputs.push_back(dbeta);
    // running_mean 和 running_var 不需要梯度
    outputs.push_back(Tensor::zeros(this->inputs_[3].shape(), dtype(DataType::kFloat32).device(x.device())));
    outputs.push_back(Tensor::zeros(this->inputs_[4].shape(), dtype(DataType::kFloat32).device(x.device())));
    return outputs;
}
}  // namespace origin
