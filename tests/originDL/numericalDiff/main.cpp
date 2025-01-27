#include "originDL.h"

using namespace dl;

Variable f(const Variable &x)
{
    auto A = Square();
    auto B = Exp();
    auto C = Square();
    return C(B(A(x)));
}

int main()
{
    auto A = Square();
    auto B = Exp();
    auto C = Square();

    auto x  = Variable(NdArray({0.5}));
    auto dy = NumericalDiff(f, x);  
    print(dy);

    return 0;
}
