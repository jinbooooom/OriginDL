#include "dlVariable.h"

namespace dl
{
Variable::Variable(const NdArray &data) : data(data) {}

Variable::~Variable() {}

void Variable::SetCreator(const FunctionPtr &func)
{
    creator = func;
}

void Variable::Backward()
{
    auto funcs = std::vector<FunctionPtr>({this->creator});
    while (!funcs.empty())
    {
        auto f = funcs.back();
        funcs.pop_back();
        auto x  = f->input;
        auto y  = f->output;
        x->grad = std::make_shared<NdArray>(f->Backward(*y->grad));

        if (x->creator != nullptr)
        {
            funcs.push_back(x->creator);
        }
    }

    return;
}

void Variable::Print()
{
    af::print("", data);
};
}  // namespace dl
