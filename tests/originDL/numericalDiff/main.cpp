#include "originDL.h"

using namespace dl;

VariablePtr f(const VariablePtr &x)
{
    auto A = std::make_shared<Square>();
    auto B = std::make_shared<Exp>();
    auto C = std::make_shared<Square>();
    return (*C)((*B)((*A)(x)));
}

int main()
{
    auto A = std::make_shared<Square>();
    auto B = std::make_shared<Exp>();
    auto C = std::make_shared<Square>();

    double val = 0.5;
    auto x     = std::make_shared<Variable>(af::constant(val, 1));
    auto dy    = NumericalDiff(
        [](Variable x) -> Variable {
            auto A      = std::make_shared<Square>();
            auto B      = std::make_shared<Exp>();
            auto C      = std::make_shared<Square>();
            auto x_ptr  = std::make_shared<Variable>(x.data);
            auto result = (*C)((*B)((*A)(x_ptr)));
            return Variable(result->data);
        },
        *x);
    print(dy);

    return 0;
}
