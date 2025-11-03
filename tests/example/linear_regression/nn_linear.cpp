#include <iostream>
#include "origin.h"
#include "origin/core/operator.h"
#include "origin/nn/layers/linear.h"
#include "origin/nn/sequential.h"
#include "origin/optim/sgd.h"
#include "origin/utils/log.h"

using namespace origin;

/**
 * @brief Neural network linear regression training demo
 * @details Uses the same data as linear_regression.cpp to verify the framework
 */
int main()
{
    // 1. Creating training data
    size_t input_size = 100;
    auto x            = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32));
    auto noise        = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32)) * 0.1f;
    auto y            = x * 2.0f + 5.0f + noise;

    // 2. Creating model
    Sequential model;
    model.add(std::make_unique<Linear>(1, 1, true));

    // 3. Creating optimizer
    float learning_rate = 0.1f;
    SGD optimizer(model, learning_rate);

    // 4. Starting training
    int iters = 200;

    model.train();

    for (int i = 0; i < iters; ++i)
    {
        optimizer.zero_grad();
        auto y_pred = model(x);  // 等价于 auto y_pred = model.forward(x);

        auto diff       = y_pred - y;
        auto sum_result = sum(pow(diff, 2));
        auto elements   = Tensor(diff.elements(), sum_result.shape(), DataType::kFloat32);
        auto loss       = sum_result / elements;

        loss.backward();

        optimizer.step();

        if (i % 10 == 0 || i == iters - 1)
        {
            float loss_val = loss.item<float>();

            // 直接通过 Linear 层访问参数，确保顺序正确
            // Sequential 的第一个模块是 Linear 层
            auto &linear_layer = dynamic_cast<Linear &>(model[0]);
            float w_val = 0.0f, b_val = 0.0f;

            w_val = linear_layer.weight()->item<float>();
            if (linear_layer.bias() != nullptr)
            {
                b_val = linear_layer.bias()->item<float>();
            }

            logi("iter{}: loss = {}, w = {}, b = {}", i, loss_val, w_val, b_val);
        }
    }

    return 0;
}
