#include <arrayfire.h>
#include <getopt.h>
#include <iostream>
#include "originDL.h"

using namespace dl;

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

VariablePtr Predict(const VariablePtr &x, const VariablePtr &w, const VariablePtr &b)
{
    auto y = matMul(x, w) + b;
    return y;
}

// mean_squared_error
VariablePtr MSE(const VariablePtr &x0, const VariablePtr &x1)
{
    auto diff = x0 - x1;
    return sum(pow(diff, 2)) / diff->data.elements();
}
int main(int argc, char **argv)
{
    SetBackend(argc, argv);

    // 设置随机种子
    af::setSeed(0);

    // 生成随机数据
    af::array xData = af::randu(100, 1);
    af::print("xData", xData);
    af::array yData = 2.0 * xData;
    // af::array yData = 2.0 * xData + 5.0;
    af::print("yData", yData);

    // 转换为变量
    auto x = AsVariablePtr(AsDLArrayPtr(xData));
    auto y = AsVariablePtr(AsDLArrayPtr(yData));

    // 初始化权重和偏置
    auto w = AsVariablePtr(AsDLArrayPtr(af::constant(100.0, 1, 1, f32)));
    auto b = AsVariablePtr(AsDLArrayPtr(af::constant(100.0, 1, 1, f32)));

    // 设置学习率和迭代次数
    double lr = 0.1;
    int iters = 100;

    // 训练
    for (int i = 0; i < iters; i++)
    {
        auto yPred = matMul(x, w);  // Predict(x, w, b);
        auto loss   = MSE(y, yPred);

        w->ClearGrad();
        b->ClearGrad();

        // 反向传播
        loss->Backward();

        // 更新参数
        w->data = w->data - lr * (*w->grad);
        // b->data = b->data - lr * (*b->grad);

        // 打印结果
        float loss_val = loss->data.scalar<float>();
        float w_val    = w->data.scalar<float>();
        float b_val    = b->data.scalar<float>();

        logi("iter{}: loss = {}, w = {}, b = {}", i, loss_val, w_val, b_val);
    }

    return 0;
}
