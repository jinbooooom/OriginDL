#include "originDL.h"

using namespace dl;

int main()
{
    auto A = std::make_shared<Square>();
    auto B = std::make_shared<Exp>();
    auto C = std::make_shared<Square>();

    double val0  = 2;
    double val1  = 3;
    af::dim4 dim = {2, 2};

    auto x0 = std::make_shared<Variable>(af::constant(val0, dim));
    auto x1 = std::make_shared<Variable>(af::constant(val1, dim));
    auto y  = x0 * x1;
    y->Backward();
    print("y", y->data);
    print("dx0", *x0->grad);
    print("dx1", *x1->grad);

    return 0;
}
