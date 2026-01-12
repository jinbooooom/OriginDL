#include <getopt.h>
#include "origin.h"

using namespace origin;
namespace F = origin::functional;
namespace nn = origin::nn;

void usage(const char *program_name)
{
    loga("Usage: {} [-d device_id] [-h]", program_name);
    loga("  -d, --device    Device ID: -1 for CPU (default), >= 0 for GPU device id");
    loga("  -h, --help      Show this help message");
}

/**
 * @brief Neural network linear regression training demo
 * @details Supports both CPU and GPU (CUDA) devices via -d option
 *          -d -1: use CPU (default)
 *          -d 0, 1, 2...: use GPU with specified device id
 */
int main(int argc, char **argv)
{
    int device_id = -1;  // 默认使用 CPU

    // 定义命令行选项
    static struct option long_options[] = {
        {"device", required_argument, 0, 'd'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "d:h", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'd':
                device_id = std::atoi(optarg);
                break;
            case 'h':
                usage(argv[0]);
                std::exit(0);
            case '?':
                loga("Use -h or --help for usage information");
                std::exit(1);
            default:
                break;
        }
    }

    // 确定设备类型
    Device device(DeviceType::kCPU);
    bool use_gpu = (device_id >= 0);

    if (use_gpu)
    {
        if (!cuda::is_available())
        {
            loge("CUDA is not available on this system!");
            return 1;
        }

        int device_count = cuda::device_count();
        if (device_id >= device_count)
        {
            loge("Invalid GPU device ID: {}. Available devices: 0-{}", device_id, device_count - 1);
            return 1;
        }

        device = Device(DeviceType::kCUDA, device_id);
        cuda::set_device(device_id);
        cuda::device_info();
        loga("Using GPU device: {}", device_id);
    }
    else
    {
        loga("Using CPU device");
    }

    // 1. Creating training data
    size_t input_size = 100;
    Tensor x, noise, y;

    if (use_gpu)
    {
        x     = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(device));
        noise = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(device)) * 0.1f;
        y     = x * 2.0f + 5.0f + noise;
    }
    else
    {
        x     = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32));
        noise = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32)) * 0.1f;
        y     = x * 2.0f + 5.0f + noise;
    }

    // 2. Creating model
    Sequential model;
    model.add(std::make_unique<nn::Linear>(1, 1, true));

    // 3. Creating optimizer (before moving to GPU, so optimizer can collect parameters)
    float learning_rate = 0.1f;
    SGD optimizer(model, learning_rate);

    // 4. Move model to device (if GPU)
    if (use_gpu)
    {
        model.to(device);
    }

    // 5. Starting training
    int iters = 200;

    model.train();

    for (int i = 0; i < iters; ++i)
    {
        optimizer.zero_grad();
        auto y_pred = model(x);

        auto diff       = y_pred - y;
        auto sum_result = F::sum(F::pow(diff, Scalar(2)));

        Tensor loss;
        if (use_gpu)
        {
            // Create elements tensor on GPU
            auto elements_value  = static_cast<float>(diff.elements());
            auto elements_tensor = Tensor(elements_value, sum_result.shape(), dtype(DataType::kFloat32).device(device));
            loss                 = sum_result / elements_tensor;
        }
        else
        {
            auto elements = Tensor(diff.elements(), sum_result.shape(), DataType::kFloat32);
            loss          = sum_result / elements;
        }

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

    return 0;
}
