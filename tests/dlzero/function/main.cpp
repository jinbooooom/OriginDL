#include "dlZero.h"

using namespace dl;

int main()
{
    auto x = Variable(NdArray({10.0}));
    auto f = Square();
    auto y = f(x);

    y.Print();

    return 0;
}
