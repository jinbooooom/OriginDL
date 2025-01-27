#include "dlZero.h"

int main()
{
    dl::NdArray datax = {1.0};
    auto x            = dl::Variable(datax);
    x.Print();

    dl::NdArray datay = {1.01, 2.01, 3.01};
    auto y            = dl::Variable(datay);
    y.Print();

    dl::NdArray dataz = {{1.0, 2.0, 3.0}, {4., 5., 6.}};
    auto z            = dl::Variable(dataz);
    z.Print();

    return 0;
}
