#include <getopt.h>
#include "origin.h"

using namespace origin;
namespace F  = origin::functional;
namespace nn = origin::nn;

void usage(const char *program_name)
{
    loga("Usage: {} [--cpu] [-d device_id] [-h]", program_name);
    loga("  --cpu           Use CPU (overrides auto/CUDA)");
    loga("  -d, --device    Device: -1 for CPU, >= 0 for GPU id. Omit for auto (CUDA if available)");
    loga("  -h, --help      Show this help message");
}

/**
 * @brief Neural network linear regression training demo
 * @details Device: auto (CUDA if available else CPU), or --cpu / -d -1 for CPU, -d N for GPU id
 */
int main(int argc, char **argv)
{
    // -2 = auto (prefer CUDA if available), -1 = CPU, >= 0 = GPU device id
    int device_id = -2;

    static struct option long_options[] = {
        {"cpu", no_argument, 0, 'c'},
        {"device", required_argument, 0, 'd'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "cd:h", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'c':
                device_id = -1;
                break;
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

    // Auto: prefer CUDA if available (like yolov5_infer)
    if (device_id == -2)
    {
        if (cuda::is_available())
        {
            device_id = 0;
        }
        else
        {
            device_id = -1;
        }
    }

    Device device(DeviceType::kCPU);
    DataType data_type = DataType::kFloat32;
    bool use_gpu       = (device_id >= 0);

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
    }

    loga("Use Device: {}", device.to_string());

    // 1. Creating training data
    size_t input_size = 100;
    Tensor x, noise, y;

    x     = Tensor::randn(Shape{input_size, 1}, dtype(data_type).device(device));
    noise = Tensor::randn(Shape{input_size, 1}, dtype(data_type).device(device)) * 0.1f;
    y     = x * 2.0f + 5.0f + noise;

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
        Tensor loss     = sum_result / static_cast<float>(diff.elements());
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

            loga("iter{}: loss = {}, w = {}, b = {}", i, loss_val, w_val, b_val);
        }
    }

    return 0;
}
