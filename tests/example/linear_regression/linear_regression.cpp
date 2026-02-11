#include <getopt.h>
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

void usage(const char *program_name)
{
    loga("Usage: {} [-d device_id] [-h]", program_name);
    loga("  -d, --device    Device ID: -1 for CPU (default), >= 0 for GPU device id");
    loga("  -h, --help      Show this help message");
}

Tensor Predict(const Tensor &x, const Tensor &w, const Tensor &b)
{
    auto y = F::mat_mul(x, w) + b;
    return y;
}

// mean_squared_error
Tensor MSE(const Tensor &x0, const Tensor &x1)
{
    auto diff       = x0 - x1;
    auto sum_result = F::sum(F::pow(diff, Scalar(2.0f)));
    auto result     = sum_result / static_cast<float>(diff.elements());  // 强转为float类型，避免类型提升。
    return result;
}

int main(int argc, char **argv)
{
    int device_id = -1;  // 默认使用 CPU

    // 定义命令行选项
    static struct option long_options[] = {
        {"device", required_argument, 0, 'd'}, {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}};

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

    // 生成随机数据
    size_t input_size  = 100;
    DataType data_type = DataType::kFloat32;
    auto x             = Tensor::randn(Shape{input_size, 1}, dtype(data_type).device(device));
    // 设置一个噪声，使真实值在预测结果附近
    auto noise = Tensor::randn(Shape{input_size, 1}, dtype(data_type).device(device)) * 0.1f;
    auto y     = x * 2.0f + 5.0f + noise;

    // 初始化权重和偏置 - 确保使用float类型以匹配输入数据
    auto w = Tensor(0.0f, Shape{1, 1}, dtype(data_type).device(device));
    auto b = Tensor(0.0f, Shape{1, 1}, dtype(data_type).device(device));

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
        if (i % 10 == 0 || i == iters - 1) 
        {
            float loss_val = loss.item<float>();
            float w_val    = w.item<float>();
            float b_val    = b.item<float>();

            loga("iter{}: loss = {}, w = {}, b = {}", i, loss_val, w_val, b_val);
        }
    }

    return 0;
}
