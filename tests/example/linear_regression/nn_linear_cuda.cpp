#include <iostream>
#include "origin.h"
#include "origin/core/operator.h"
#include "origin/nn/layers/linear.h"
#include "origin/nn/sequential.h"
#include "origin/optim/sgd.h"
#include "origin/utils/log.h"

using namespace origin;

namespace nn = origin::nn;

/**
 * @brief Neural network linear regression training demo (CUDA version)
 * @details Uses the same data as linear_regression.cpp to verify the framework with GPU
 */
int main()
{
#ifdef WITH_CUDA
    cuda::print_cuda_device_info();

    if (!cuda::is_cuda_available())
    {
        std::cout << "CUDA is not available on this system!" << std::endl;
        return 1;
    }

    // 1. Creating training data on GPU
    size_t input_size = 100;
    auto x            = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(kCUDA));
    auto noise        = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(kCUDA)) * 0.1f;
    auto y            = x * 2.0f + 5.0f + noise;

    // 2. Creating model
    Sequential model;
    model.add(std::make_unique<nn::Linear>(1, 1, true));

    // 3. Creating optimizer (before moving to GPU, so optimizer can collect parameters)
    float learning_rate = 0.1f;
    SGD optimizer(model, learning_rate);

    // 4. Move model to GPU (after optimizer is created)
    model.to(Device(DeviceType::kCUDA));

    // 5. Starting training
    int iters = 200;

    model.train();

    for (int i = 0; i < iters; ++i)
    {
        optimizer.zero_grad();

        auto y_pred = model(x);

        auto diff       = y_pred - y;
        auto sum_result = sum(pow(diff, 2));
        // Create elements tensor on GPU (scalar with value = diff.elements())
        auto elements_value  = static_cast<float>(diff.elements());
        auto elements_tensor = Tensor(elements_value, sum_result.shape(), dtype(DataType::kFloat32).device(kCUDA));
        auto loss            = sum_result / elements_tensor;

        loss.backward();

        optimizer.step();

        if (i % 10 == 0 || i == iters - 1)
        {
            float loss_val = loss.item<float>();

            // 直接通过 Linear 层访问参数，确保顺序正确
            auto &linear_layer = dynamic_cast<nn::Linear &>(model[0]);
            float w_val = 0.0f, b_val = 0.0f;

            w_val = linear_layer.weight()->item<float>();
            if (linear_layer.bias() != nullptr)
            {
                b_val = linear_layer.bias()->item<float>();
            }

            logi("iter{}: loss = {}, w = {}, b = {}", i, loss_val, w_val, b_val);
        }
    }

#else
    std::cout << "CUDA support is not enabled in this build!" << std::endl;
    std::cout << "Please rebuild with --cuda flag: ./build.sh origin --cuda" << std::endl;
    return 1;
#endif

    return 0;
}
