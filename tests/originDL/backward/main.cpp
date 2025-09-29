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
    data_t val   = 0.5;

    logi("Test: y = (exp(x^2))^2");
    auto x = Tensor::constant(val, Shape{static_cast<size_t>(dim[0]), static_cast<size_t>(dim[1])});
    auto a = (*A)(x);
    auto b = (*B)(a);
    auto y = (*C)(b);

    y.backward();
    x.grad().print("gx: ");  // gx = 3.2974

    logi("Test Add: y = (x0^2) + (x1^2)");
    auto x0 = Tensor::constant(2, Shape{static_cast<size_t>(dim[0]), static_cast<size_t>(dim[1])});
    auto x1 = Tensor::constant(3, Shape{static_cast<size_t>(dim[0]), static_cast<size_t>(dim[1])});
    // y       = add({square(x0), square(x1)});
    // y = (x0) ^ 2 + (x1) ^ 2; 被解释成 y = ((x0) ^ (2 + (x1))) ^ 2; 与预期不符
    // 原本的 ^ 指的是异或，运算符优先级比 + 要低，所以要用括号。
    y = ((x0) ^ 2) + ((x1) ^ 2);
    y.print("y: ");  // 13
    y.backward();
    x0.grad().print("gx0: ");  // 4
    x1.grad().print("gx1: ");  // 6

    logi("Test Add with repeated values: y = x + x");
    y.clear_grad();
    x = x0;
    x.clear_grad();
    y = x + x;
    y.print("y: ");  // 4
    y.backward();
    x.grad().print("gx: ");  // 2

    logi("Test Complex computation graph: y = ((x^2)^2) + ((x^2)^2) = 2 * (x^4)");
    y.clear_grad();
    x = Tensor::constant(2, Shape{static_cast<size_t>(dim[0]), static_cast<size_t>(dim[1])});
    // auto s = square(x);
    // y      = add({square(s), square(s)});
    y = ((x ^ 2) ^ 2) + ((x ^ 2) ^ 2);
    y.print("y: ");  // 32
    y.backward();
    x.grad().print("gx: ");  // 64

    // reshape
    logi("Test Reshape:");
    y.clear_grad();
    af::array tensor3_4 = af::randu(3, 4);  // 3 行 4 列随机值
    auto x3_4           = Tensor::from_mat_for_test(std::make_unique<Mat_t>(tensor3_4));
    x3_4.print("before reshape, x: ");
    const Shape shape_3{4, 3};
    y = dl::reshape(x3_4, shape_3);
    y.backward();
    y.print("after reshape, y: ");
    x3_4.grad().print("gx: ");

    // transpose
    logi("Test Transpose:");
    y.clear_grad();
    x3_4.clear_grad();
    tensor3_4 = af::randu(3, 4);  // 3 行 4 列随机值
    x3_4      = Tensor::from_mat_for_test(std::make_unique<Mat_t>(tensor3_4));
    x3_4.print("before reshape, x: ");
    y = dl::transpose(x3_4);
    y.backward();
    y.print("after transpose, y: ");
    x3_4.grad().print("gx: ");

    // sum
    logi("Test Sum:");
    y.clear_grad();
    af::array tensor2_4 = af::iota(af::dim4(2, 4));
    auto x2_4           = Tensor::from_mat_for_test(std::make_unique<Mat_t>(tensor2_4));
    x2_4.print("before sum, x: ");
    y = dl::sum(x2_4);
    y.backward();
    y.print("after sum, y: ");
    x2_4.grad().print("gx: ");

    // sumTo
    logi("Test SumTo:");
    y.clear_grad();
    x2_4.clear_grad();
    x2_4.print("before sumTo, x: ");
    y = dl::sum_to(x2_4, Shape{1, 4});
    y.backward();
    y.print("after sumTo, y: ");
    x2_4.grad().print("gx: ");

    // broadcastTo
    logi("Test BroadcastTo:");
    y.clear_grad();
    af::array tensor1_4 = af::iota(af::dim4(1, 4));
    auto x1_4           = Tensor::from_mat_for_test(std::make_unique<Mat_t>(tensor1_4));
    x1_4.clear_grad();
    x1_4.print("before broadcastTo, x: ");
    y = dl::broadcast_to(x1_4, Shape{2, 4});
    y.backward();
    y.print("after broadcastTo, y: ");
    x1_4.grad().print("gx: ");

    // matMul
    logi("Test matMul:");
    {
        af::array tensor_x = af::iota(af::dim4(2, 4));
        auto x             = Tensor::from_mat_for_test(std::make_unique<Mat_t>(tensor_x));
        af::array tensor_w = af::iota(af::dim4(4, 2));
        auto w             = Tensor::from_mat_for_test(std::make_unique<Mat_t>(tensor_w));
        x.print("before matMul, X: ");
        w.print("before matMul, W: ");
        auto y = dl::mat_mul(x, w);
        y.backward();
        y.print("after matMul, y: ");
        x.grad().print("gx: ");
        w.grad().print("gw: ");
    }

    return 0;
}
