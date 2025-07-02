#include "dlFunction.h"

namespace dl
{

VariablePtrList Function::operator()(const VariablePtr &input)
{
    auto outputs = (*this)(VariablePtrList({input}));
    return outputs;
}

VariablePtrList Function::operator()(const VariablePtrList &inputs)
{
    auto xs = NdArrayPtrList();
    for (const auto &i : inputs)
    {
        xs.push_back(AsDLArrayPtr(i->data));
    }

    auto ys      = this->Forward(xs);
    auto outputs = VariablePtrList();
    for (const auto &y : ys)
    {
        auto o = AsVariablePtr(y);
        o->SetCreator(shared_from_this());
        outputs.push_back(o);
    }

    int maxGen = 0;
    for (auto &e : inputs)
    {
        if (e->generation > maxGen)
        {
            maxGen = e->generation;
        }
    }
    this->generation = maxGen;

    this->inputs  = inputs;
    this->outputs = std::move(outputs);
    return this->outputs;
}

}  // namespace dl
