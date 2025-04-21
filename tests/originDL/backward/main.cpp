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

	double val = 0.5;
	auto x = std::make_shared<Variable>(af::constant(val, 1));
	auto a = (*A)(x);
	auto b = (*B)(a);
	auto y = (*C)(b);

	double grad_val = 1.0;
	y->grad = std::make_shared<NdArray>(af::constant(grad_val, 1));
	y->Backward();
    print(*x->grad); 
	// std::cout << NdArrayPrinter(x->grad) << std::endl;

    return 0;
}
