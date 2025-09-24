#include "originDL.h"

using namespace dl;

int main()
{
    double val0     = 2;
    double val1     = 4;
    af::dim4 dim    = {2, 2};
    auto x0         = std::make_shared<Variable>(af::constant(val0, dim));
    auto x1         = std::make_shared<Variable>(af::constant(val1, dim));
    auto y          = -x0;
    auto clear_grad = [&]() {
        y->clear_grad();
        x0->clear_grad();
        x1->clear_grad();
    };

    logi("Neg: y = -x0");
    clear_grad();
    y = -x0;
    y->backward();
    print("y", y->data_);
    print("dx0", *x0->grad_);

    logi("Add: y = x0 + x1");
    clear_grad();
    y = x0 + x1;
    y->backward();
    print("y", y->data_);
    print("dx0", *x0->grad_);
    print("dx1", *x1->grad_);

    logi("Sub: y = x0 - x1");
    clear_grad();
    y = x0 - x1;
    y->backward();
    print("y", y->data_);
    print("dx0", *x0->grad_);
    print("dx1", *x1->grad_);

    logi("Mul: y = x0 * x1");
    clear_grad();
    y = x0 * x1;
    y->backward();
    print("y", y->data_);
    print("dx0", *x0->grad_);
    print("dx1", *x1->grad_);

    logi("Div: y = x0 / x1");
    clear_grad();
    y = x0 / x1;
    y->backward();
    print("y", y->data_);
    print("dx0", *x0->grad_);  // 1 / x1
    print("dx1", *x1->grad_);  // -x0 / x1^2

    logi("Square: y = x0^2");
    clear_grad();
    y = square(x0);
    y->backward();
    print("y", y->data_);
    print("dx0", *x0->grad_);

    logi("Pow: y = x0^3");
    clear_grad();
    y = x0 ^ 3;
    y->backward();
    print("y", y->data_);
    print("dx0", *x0->grad_);

    logi("Exp: y = exp(x0)");
    clear_grad();
    y = exp(x0);
    y->backward();
    print("y", y->data_);
    print("dx0", *x0->grad_);

    return 0;
}
