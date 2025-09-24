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
    print("gx: ", *x->grad_);  // gx = 3.2974

    logi("Test Add: y = (x0^2) + (x1^2)");
    auto x0 = std::make_shared<Variable>(af::constant(2, dim));
    auto x1 = std::make_shared<Variable>(af::constant(3, dim));
    // y       = add({square(x0), square(x1)});
    // y = (x0) ^ 2 + (x1) ^ 2; 被解释成 y = ((x0) ^ (2 + (x1))) ^ 2; 与预期不符
    // 原本的 ^ 指的是异或，运算符优先级比 + 要低，所以要用括号。
    y = ((x0) ^ 2) + ((x1) ^ 2);
    print("y:", y->data_);  // 13
    y->Backward();
    print("gx0:", *(x0->grad_));  // 4
    print("gx1:", *(x1->grad_));  // 6

    logi("Test Add with repeated values: y = x + x");
    y->ClearGrad();
    x = x0;
    x->ClearGrad();
    y = x + x;
    print("y:", y->data_);  // 4
    y->Backward();
    print("gx:", *(x->grad_));  // 2

    logi("Test Complex computation graph: y = ((x^2)^2) + ((x^2)^2) = 2 * (x^4)");
    y->ClearGrad();
    x = std::make_shared<Variable>(af::constant(2, dim));
    // auto s = square(x);
    // y      = add({square(s), square(s)});
    y = ((x ^ 2) ^ 2) + ((x ^ 2) ^ 2);
    print("y:", y->data_);  // 32
    y->Backward();
    print("gx:", *(x->grad_));  // 64

    // reshape
    logi("Test Reshape:");
    y->ClearGrad();
    af::array tensor3_4 = af::randu(3, 4);  // 3 行 4 列随机值
    auto x3_4           = std::make_shared<Variable>(tensor3_4);
    print("before reshape, x:", x3_4->data_);
    const af::dim4 dim4_3{4, 3};
    y = reshape(x3_4, dim4_3);
    y->Backward();
    print("after reshape, y:", y->data_);
    print("gx:", *(x3_4->grad_));

    // transpose
    logi("Test Transpose:");
    y->ClearGrad();
    x3_4->ClearGrad();
    tensor3_4 = af::randu(3, 4);  // 3 行 4 列随机值
    x3_4      = std::make_shared<Variable>(tensor3_4);
    print("before reshape, x:", x3_4->data_);
    y = transpose(x3_4);
    y->Backward();
    print("after transpose, y:", y->data_);
    print("gx:", *(x3_4->grad_));

    // sum
    logi("Test Sum:");
    y->ClearGrad();
    af::array tensor2_4 = af::iota(af::dim4(2, 4));
    auto x2_4           = std::make_shared<Variable>(tensor2_4);
    print("before sum, x:", x2_4->data_);
    y = sum(x2_4);
    y->Backward();
    print("after sum, y:", y->data_);
    print("gx:", *(x2_4->grad_));

    // sumTo
    logi("Test SumTo:");
    y->ClearGrad();
    x2_4->ClearGrad();
    print("before sumTo, x:", x2_4->data_);
    y = sumTo(x2_4, af::dim4(1, 4));
    y->Backward();
    print("after sumTo, y:", y->data_);
    print("gx:", *(x2_4->grad_));

    // broadcastTo
    logi("Test BroadcastTo:");
    y->ClearGrad();
    af::array tensor1_4 = af::iota(af::dim4(1, 4));
    auto x1_4           = std::make_shared<Variable>(tensor1_4);
    x1_4->ClearGrad();
    print("before broadcastTo, x:", x1_4->data_);
    y = broadcastTo(x1_4, af::dim4(2, 4));
    y->Backward();
    print("after broadcastTo, y:", y->data_);
    print("gx:", *(x1_4->grad_));

    // matMul
    logi("Test matMul:");
    {
        af::array tensorX = af::iota(af::dim4(2, 4));
        auto x            = std::make_shared<Variable>(tensorX);
        af::array tensorW = af::iota(af::dim4(4, 2));
        auto w            = std::make_shared<Variable>(tensorW);
        print("before matMul, X:", x->data_);
        print("before matMul, W:", w->data_);
        auto y = matMul(x, w);
        y->Backward();
        print("after matMul, y:", y->data_);
        print("gx:", *(x->grad_));
        print("gw:", *(w->grad_));
    }

    return 0;
}
