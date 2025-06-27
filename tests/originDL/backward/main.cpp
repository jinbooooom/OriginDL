#include "originDL.h"

using namespace dl;

int main()
{
    // auto A = Square();
    // auto B = Exp();
    // auto C = Square();

    // auto x = Variable(NdArray({0.5}));
    // auto a = A(x);
    // auto b = B(a);
    // auto y = C(b);

    // y.grad = std::make_shared<NdArray>(NdArray({1.0}));
    // b.grad = std::make_shared<NdArray>(C.Backward(*y.grad));
    // a.grad = std::make_shared<NdArray>(B.Backward(*b.grad));
    // x.grad = std::make_shared<NdArray>(A.Backward(*a.grad));
    // print(*x.grad);

    auto A = FunctionPtr(new Square());
    auto B = FunctionPtr(new Exp());
    auto C = FunctionPtr(new Square());

    af::dim4 dim = {2,2};
    double val = 0.5;
    auto x     = std::make_shared<Variable>(af::constant(val, dim));
    auto a     = (*A)(x);
    auto b     = (*B)(a);
    auto y     = (*C)(b);

    // double grad_val = 1.0;
    // y->grad         = std::make_shared<NdArray>(af::constant(grad_val, dim));
    y->Backward();
    print(*x->grad);

    return 0;
}
