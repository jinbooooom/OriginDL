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

    // double grad_val = 1.0;
    // y->grad         = std::make_shared<NdArray>(af::constant(grad_val, dim));
    y->Backward();
    print(*x->grad);

    logi("Test Add");
    auto add  = FunctionPtr(new Add());
    auto x0   = std::make_shared<Variable>(af::constant(2, dim));
    auto x1   = std::make_shared<Variable>(af::constant(3, dim));
    auto yAdd = (*add)({x0, x1});
    print((yAdd[0])->data);

    return 0;
}
