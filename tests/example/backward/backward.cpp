#include <getopt.h>
#include <iostream>
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

int main(int argc, char **argv)
{
    origin::Shape shape = {2, 2};

    logi("Test Add: y = (x0^2) + (x1^2)");
    auto x0 = origin::Tensor(2, shape);
    auto x1 = origin::Tensor(3, shape);
    // y       = origin::add({origin::square(x0), origin::square(x1)});
    // y = (x0) ^ 2 + (x1) ^ 2; 被解释成 y = ((x0) ^ (2 + (x1))) ^ 2; 与预期不符
    // 原本的 ^ 指的是异或，运算符优先级比 + 要低，所以要用括号。
    auto y = ((x0) ^ 2) + ((x1) ^ 2);
    y.print("y: ");  // 13
    y.backward();
    x0.grad().print("gx0: ");  // 4
    x1.grad().print("gx1: ");  // 6

    logi("Test Add with repeated values: y = x + x");
    y.clear_grad();
    auto x = x0;
    x.clear_grad();
    y = x + x;
    y.print("y: ");  // 4
    y.backward();
    x.grad().print("gx: ");  // 2

    logi("Test Complex computation graph: y = ((x^2)^2) + ((x^2)^2) = 2 * (x^4)");
    y.clear_grad();
    x = origin::Tensor(2, shape);
    // auto s = origin::square(x);
    // y      = origin::add({origin::square(s), origin::square(s)});
    y = ((x ^ 2) ^ 2) + ((x ^ 2) ^ 2);
    y.print("y: ");  // 32
    y.backward();
    x.grad().print("gx: ");  // 64

    // reshape
    logi("Test Reshape:");
    y.clear_grad();
    // auto x3_4 = origin::Tensor::randn(origin::Shape{3, 4});  // 3 行 4 列随机值
    auto x3_4 = origin::Tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, origin::Shape{3, 4});
    x3_4.print("before reshape, x: ");
    const origin::Shape shape_3{4, 3};
    y = F::reshape(x3_4, shape_3);
    y.backward();
    y.print("after reshape, y: ");
    x3_4.grad().print("gx: ");

    // transpose
    logi("Test Transpose:");
    y.clear_grad();
    x3_4.clear_grad();
    x3_4 = origin::Tensor::randn(origin::Shape{3, 4}, origin::dtype(origin::DataType::kFloat32));  // 3 行 4 列随机值
    x3_4.print("before reshape, x: ");
    y = F::transpose(x3_4);
    y.backward();
    y.print("after transpose, y: ");
    x3_4.grad().print("gx: ");

    // sum
    logi("Test Sum:");
    y.clear_grad();
    auto x2_4 = origin::Tensor({0, 1, 2, 3, 4, 5, 6, 7}, origin::Shape{2, 4}, origin::Float32);
    x2_4.print("before sum, x: ");
    y = F::sum(x2_4);
    y.backward();
    y.print("after sum, y: ");
    x2_4.grad().print("gx: ");

    // sumTo
    logi("Test SumTo:");
    y.clear_grad();
    x2_4.clear_grad();
    x2_4.print("before sumTo, x: ");
    y = F::sum_to(x2_4, origin::Shape{1, 4});
    y.backward();
    y.print("after sumTo, y: ");
    x2_4.grad().print("gx: ");

    // broadcastTo
    logi("Test BroadcastTo:");
    y.clear_grad();
    auto x1_4 = origin::Tensor({0, 1, 2, 3}, origin::Shape{1, 4}, origin::Float32);
    x1_4.clear_grad();
    x1_4.print("before broadcastTo, x: ");
    y = F::broadcast_to(x1_4, origin::Shape{2, 4});
    y.backward();
    y.print("after broadcastTo, y: ");
    x1_4.grad().print("gx: ");

    // matMul
    logi("Test matMul:");
    {
        // auto vx = std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
        // auto vw = std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
        // auto x = origin::Tensor(vx, origin::Shape{2, 4}, origin::Float32);
        // auto w = origin::Tensor(vw, origin::Shape{4, 2}, origin::Float32);
        auto x = origin::Tensor({0, 1, 2, 3, 4, 5, 6, 7}, origin::Shape{2, 4}, origin::Float32);
        auto w = origin::Tensor({0, 1, 2, 3, 4, 5, 6, 7}, origin::Shape{4, 2}, origin::Float32);
        x.print("before matMul, X: ");
        w.print("before matMul, W: ");
        auto y = F::mat_mul(x, w);
        y.backward();
        y.print("after matMul, y: ");
        x.grad().print("gx: ");
        w.grad().print("gw: ");
    }

    return 0;
}
