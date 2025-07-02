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

int main(int argc, char **argv)
{
    SetBackend(argc, argv);

    auto A = FunctionPtr(new Square());
    auto B = FunctionPtr(new Exp());
    auto C = FunctionPtr(new Square());

    af::dim4 dim = {2, 2};
    double val   = 0.5;

    logi("Test: y = (exp(x^2))^2");
    auto x = std::make_shared<Variable>(af::constant(val, dim));
    auto a = (*A)(x)[0];
    auto b = (*B)(a)[0];
    auto y = (*C)(b)[0];

    y->Backward();
    print("gx: ", *x->grad);  // gx = 3.2974

    logi("Test Add: y = x0^2 + x1^2");
    auto x0 = std::make_shared<Variable>(af::constant(2, dim));
    auto x1 = std::make_shared<Variable>(af::constant(3, dim));
    y       = add({square(x0), square(x1)});
    print("y:", y->data);  // 13
    y->Backward();
    print("gx0:", *(x0->grad));  // 4
    print("gx1:", *(x1->grad));  // 6

    logi("Test Add with repeated values: y = x + x");
    y->ClearGrad();
    x = x0;
    x->ClearGrad();
    y = add({x, x});
    print("y:", y->data);  // 4
    y->Backward();
    print("gx:", *(x->grad));  // 2

    logi("Test Complex computation graph: y = (x^2)^2 + (x^2)^2 = 2 * x^4");
    y->ClearGrad();
    x      = std::make_shared<Variable>(af::constant(2, dim));
    auto s = square(x);
    y      = add({square(s), square(s)});
    print("y:", y->data);  // 32
    y->Backward();
    print("gx:", *(x->grad));  // 64

    return 0;
}
