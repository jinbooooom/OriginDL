#include "originDL.h"

using namespace dl;

int main()
{
    auto A = FunctionPtr(new Square());
    auto B = FunctionPtr(new Exp());
    auto C = FunctionPtr(new Square());

    af::dim4 dim = {2, 2};
    double val   = 0.5;
    auto x       = std::make_shared<Variable>(af::constant(val, dim));
    auto a       = (*A)(x);
    auto b       = (*B)(a);
    auto y       = (*C)(b);

    y->Backward();
    logi("y = (exp(x^2))^2");
    print("gx: ", *x->grad);  // gx = 3.2974

    logi("Test Add: ");
    auto add     = FunctionPtr(new Add());
    auto square0 = FunctionPtr(new Square());
    auto square1 = FunctionPtr(new Square());
    auto x0      = std::make_shared<Variable>(af::constant(2, dim));
    auto x1      = std::make_shared<Variable>(af::constant(3, dim));
    auto y0      = (*square0)(x0);
    auto y1      = (*square1)(x1);
    auto yAdd    = (*add)({y0, y1});
    logi("z = x0^2 + x1^2");
    auto sum = yAdd[0];
    print("z:", (sum)->data);  // 13
    sum->Backward();
    print("gx0:", *(x0->grad));  // 4
    print("gx1:", *(x1->grad));  // 6

    logi("Test Add with repeated values: ");
    for (auto y : yAdd)
    {
        y->ClearGrad();
    }
    add  = FunctionPtr(new Add());
    x    = std::make_shared<Variable>(af::constant(2, dim));
    yAdd = (*add)({x, x});
    logi("y = x + x");
    sum = yAdd[0];
    print("y:", (sum)->data);  // 4
    sum->Backward();
    print("gx:", *(x->grad));  // 2

    return 0;
}
