#include "originDL.h"

using namespace dl;

int main()
{
    // 输出设备信息并初始化ArrayFire
    // 初始化ArrayFire后端
    try
    {
        af::setBackend(AF_BACKEND_CPU);  // 使用CPU后端
        af::info();                      // 输出设备信息并初始化ArrayFire
    }
    catch (const af::exception &e)
    {
        loge("Failed to initialize ArrayFire: {}", e.what());
        return 1;
    }

    double val0     = 2;
    double val1     = 4;
    af::dim4 dim    = {2, 2};
    auto x0         = Tensor(af::constant(val0, dim));
    auto x1         = Tensor(af::constant(val1, dim));
    auto y          = -x0;
    auto clear_grad = [&]() {
        y.clear_grad();
        x0.clear_grad();
        x1.clear_grad();
    };

    logi("Neg: y = -x0");
    clear_grad();
    y = -x0;
    y.backward();
    print("y", y.data());
    print("dx0", x0.grad());

    logi("Add: y = x0 + x1");
    clear_grad();
    y = x0 + x1;
    y.backward();
    print("y", y.data());
    print("dx0", x0.grad());
    print("dx1", x1.grad());

    logi("Sub: y = x0 - x1");
    clear_grad();
    y = x0 - x1;
    y.backward();
    print("y", y.data());
    print("dx0", x0.grad());
    print("dx1", x1.grad());

    logi("Mul: y = x0 * x1");
    clear_grad();
    y = x0 * x1;
    y.backward();
    print("y", y.data());
    print("dx0", x0.grad());
    print("dx1", x1.grad());

    logi("Div: y = x0 / x1");
    clear_grad();
    y = x0 / x1;
    y.backward();
    print("y", y.data());
    print("dx0", x0.grad());  // 1 / x1
    print("dx1", x1.grad());  // -x0 / x1^2

    logi("Square: y = x0^2");
    clear_grad();
    y = square(x0);
    y.backward();
    print("y", y.data());
    print("dx0", x0.grad());

    logi("Pow: y = x0^3");
    clear_grad();
    y = x0 ^ 3;
    y.backward();
    print("y", y.data());
    print("dx0", x0.grad());

    logi("Exp: y = exp(x0)");
    clear_grad();
    y = exp(x0);
    y.backward();
    print("y", y.data());
    print("dx0", x0.grad());

    return 0;
}
