#include <getopt.h>
#include <iostream>
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

Tensor Predict(const Tensor &x, const Tensor &w, const Tensor &b)
{
    auto y = F::mat_mul(x, w) + b;
    return y;
}

// mean_squared_error
Tensor MSE(const Tensor &x0, const Tensor &x1)
{
    auto diff       = x0 - x1;
    auto sum_result = F::sum(F::pow(diff, Scalar(2)));
    // 使用除法算子而不是直接创建Tensor，确保有正确的creator_
    auto elements = Tensor(diff.elements(), sum_result.shape(), DataType::kFloat32);
    auto result   = sum_result / elements;
    return result;
}
int main(int argc, char **argv)
{
    // 生成随机数据
    size_t input_size = 100;
    auto x            = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32));
    // 设置一个噪声，使真实值在预测结果附近
    auto noise = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32)) * 0.1f;
    auto y     = x * 2.0f + 5.0f + noise;

    // 初始化权重和偏置 - 确保使用float类型以匹配输入数据
    auto w = Tensor(0.0f, Shape{1, 1});
    auto b = Tensor(0.0f, Shape{1, 1});

    // 设置学习率和迭代次数
    float lr  = 0.1f;
    int iters = 200;

    // 训练
    for (int i = 0; i < iters; i++)
    {
        // 清零梯度
        w.clear_grad();
        b.clear_grad();

        auto y_pred = Predict(x, w, b);
        auto loss   = MSE(y, y_pred);

        // 反向传播
        loss.backward();

        // 更新参数 - 使用算子而不是直接修改data()
        w = w - lr * w.grad();
        b = b - lr * b.grad();

        // 打印结果
        float loss_val = loss.to_vector<float>()[0];
        float w_val    = w.to_vector<float>()[0];
        float b_val    = b.to_vector<float>()[0];

        logi("iter{}: loss = {}, w = {}, b = {}", i, loss_val, w_val, b_val);
    }

    return 0;
}
