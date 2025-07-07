#include "originDL.h"

using namespace dl;

int main()
{
    double val0    = 2;
    double val1    = 4;
    af::dim4 dim   = {2, 2};
    auto x0        = std::make_shared<Variable>(af::constant(val0, dim));
    auto x1        = std::make_shared<Variable>(af::constant(val1, dim));
    auto y         = -x0;
    auto ClearGrad = [&]() {
        y->ClearGrad();
        x0->ClearGrad();
        x1->ClearGrad();
    };

    logi("Neg: y = -x0");
    ClearGrad();
    y = -x0;
    y->Backward();
    print("y", y->data);
    print("dx0", *x0->grad);

    logi("Add: y = x0 + x1");
    ClearGrad();
    y = x0 + x1;
    y->Backward();
    print("y", y->data);
    print("dx0", *x0->grad);
    print("dx1", *x1->grad);

    logi("Sub: y = x0 - x1");
    ClearGrad();
    y = x0 - x1;
    y->Backward();
    print("y", y->data);
    print("dx0", *x0->grad);
    print("dx1", *x1->grad);

    logi("Mul: y = x0 * x1");
    ClearGrad();
    y = x0 * x1;
    y->Backward();
    print("y", y->data);
    print("dx0", *x0->grad);
    print("dx1", *x1->grad);

    logi("Div: y = x0 / x1");
    ClearGrad();
    y = x0 / x1;
    y->Backward();
    print("y", y->data);
    print("dx0", *x0->grad);  // 1 / x1
    print("dx1", *x1->grad);  // -x0 / x1^2

    return 0;
}
