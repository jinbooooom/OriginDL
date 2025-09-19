#include "dlOperator.h"

namespace dl
{

VariablePtrList Operator::operator()(const VariablePtr &input)
{
    auto outputs = (*this)(VariablePtrList({input}));
    return outputs;
}

VariablePtrList Operator::operator()(const VariablePtrList &inputs)
{
    auto xs = NdArrayPtrList();
    for (const auto &i : inputs)
    {
        xs.push_back(AsDLArrayPtr(i->mData));
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
        if (e->mGeneration > maxGen)
        {
            maxGen = e->mGeneration;
        }
    }
    this->mGeneration = maxGen;

    this->inputs = inputs;
    // this->outputs = std::move(outputs);
    this->outputs.clear();
    for (const auto &o : outputs)
    {
        VariableWPtr w = o;
        this->outputs.push_back(w);
    }

    return outputs;
}

}  // namespace dl
