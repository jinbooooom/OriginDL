#include "dlZero.h"

using namespace dl;

int main()
{
    auto A = Square();
	auto B = Exp();
	auto C = Square();

	auto x = Variable(NdArray({ 0.5 }));
	auto a = A(x);
	auto b = B(a);
	auto y = C(b);
	print(y.data);

    return 0;
}
