#include <arrayfire.h>
#include <getopt.h>
#include <iostream>
#include "origin.h"
#include "origin/utils/log.h"

using namespace origin;

void Usage()
{
    std::cout << "Usage: ./backend_demo -b [backend_code]\n"
              << "Available backends:\n"
              << "0 : CPU\n"
              << "1 : CUDA\n"
              << "2 : OpenCL\n";
}

int SetBackend(int argc, char **argv)
{
    int backend_code = 0;  // 默认CPU
    int c;

    // 解析命令行参数
    while ((c = getopt(argc, argv, "b:h")) != -1)
    {
        switch (c)
        {
            case 'b':
                backend_code = atoi(optarg);
                break;
            case 'h':
                Usage();
                return 0;
            default:
                return 1;
        }
    }

    // 设置计算后端
    try
    {
        switch (backend_code)
        {
            case 1:
                af::setBackend(AF_BACKEND_CUDA);  // [4]() CUDA后端
                break;
            case 2:
                af::setBackend(AF_BACKEND_OPENCL);  // OpenCL后端
                break;
            default:
                af::setBackend(AF_BACKEND_CPU);  // CPU后端
        }
    }
    catch (const af::exception &e)
    {
        loge("Failed to set backend: {}", e.what());
        return 1;
    }

    // 验证后端设置
    af::info();  // 输出设备信息
    std::cout << std::endl;
    switch (af::getActiveBackend())
    {  // 获取当前后端
        case AF_BACKEND_CUDA:
            logw("Active Backend: CUDA");
            break;
        case AF_BACKEND_OPENCL:
            logw("Active Backend: OpenCL");
            break;
        case AF_BACKEND_CPU:
            logw("Active Backend: CPU");
            break;
        default:
            loge("Invalid Backend");
            exit(0);
    }

    return 0;
}

#define USE_BIAS (1)

#if USE_BIAS
Tensor Predict(const Tensor &x, const Tensor &w, const Tensor &b)
{
    auto y = origin::mat_mul(x, w) + b;
    return y;
}
#else
Tensor Predict(const Tensor &x, const Tensor &w)
{
    auto y = origin::mat_mul(x, w);
    return y;
}
#endif

// mean_squared_error
Tensor MSE(const Tensor &x0, const Tensor &x1)
{
    auto diff       = x0 - x1;
    auto sum_result = origin::sum(origin::pow(diff, 2));
    // 使用除法算子而不是直接创建Tensor，确保有正确的creator_
    auto elements = Tensor::constant(diff.elements(), sum_result.shape());
    auto result   = sum_result / elements;
    return result;
}
int main(int argc, char **argv)
{
    SetBackend(argc, argv);

    // 设置随机种子
    af::setSeed(0);

    // 生成随机数据
    size_t input_size = 100;
    auto x            = Tensor::randn(Shape{input_size, 1});
    // 设置一个噪声，使真实值在预测结果附近
    auto noise = Tensor::randn(Shape{input_size, 1}) * 0.1;
#if USE_BIAS
    auto y = x * 2.0 + 5.0 + noise;
#else
    auto y = x * 2.0 + noise;
#endif

    // 初始化权重和偏置
    auto w = Tensor::constant(0, Shape{1, 1});
#if USE_BIAS
    auto b = Tensor::constant(0, Shape{1, 1});
#endif

    // 设置学习率和迭代次数
    data_t lr = 0.1;
    int iters = 200;

    // 训练
    for (int i = 0; i < iters; i++)
    {
        // 清零梯度
        w.clear_grad();
#if USE_BIAS
        b.clear_grad();
#endif

#if USE_BIAS
        auto y_pred = Predict(x, w, b);
        auto loss   = MSE(y, y_pred);
#else
        auto y_pred = Predict(x, w);
        auto loss   = MSE(y, y_pred);
#endif

        // 反向传播
        loss.backward();

        // 更新参数 - 使用算子而不是直接修改data()
        w = w - lr * w.grad();
#if USE_BIAS
        b = b - lr * b.grad();
#endif

        // 打印结果
        float loss_val = loss.to_vector()[0];
        float w_val    = w.to_vector()[0];
#if USE_BIAS
        float b_val = b.to_vector()[0];
#endif

#if USE_BIAS
        logi("iter{}: loss = {}, w = {}, b = {}", i, loss_val, w_val, b_val);
#else
        logi("iter{}: loss = {}, w = {}", i, loss_val, w_val);
#endif
    }

    return 0;
}
