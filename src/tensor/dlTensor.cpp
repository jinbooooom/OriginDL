#include "originDL.h"

namespace dl
{
Variable::Variable(const NdArray &data) : data(data), generation(0) {}

Variable::~Variable() {}

void Variable::SetCreator(const FunctionPtr &func)
{
    creator    = func;
    generation = creator->generation + 1;
}

void Variable::Backward()
{
    if (!this->grad)
    {
        double grad_val = 1.0;
        auto dims       = this->data.dims();
        grad            = std::make_shared<NdArray>(af::constant(grad_val, dims));
    }

    auto funcs   = std::list<FunctionPtr>();
    auto funcSet = std::set<FunctionPtr>();  // 考虑到多输出的情况下

    auto AddFunc = [&funcs, &funcSet](const FunctionPtr &f) {
        if (funcSet.find(f) == funcSet.end())
        {
            funcs.push_back(f);
            funcSet.insert(f);
            funcs.sort(
                [](const FunctionPtr &lhs, const FunctionPtr &rhs) { return lhs->generation < rhs->generation; });
        }
    };

    AddFunc(this->creator);

    while (!funcs.empty())
    {
        auto f = funcs.back();
        funcs.pop_back();

        auto gys = NdArrayPtrList();
        for (const auto &o : f->outputs)
        {
            // 通过 lock() 升级为 shared_ptr 并检查有效性
            if (auto oPtr = o.lock())
            {
                gys.emplace_back(oPtr->grad);
            }
            else
            {
                loge("backward error!, output is nullptr");
                exit(0);
            }
        }
        auto gxs = f->Backward(gys);

        if (gxs.size() != f->inputs.size())
        {
            loge("backward error!, gxs size {}, inputs size {}", gxs.size(), f->inputs.size());
            exit(1);
        }

        for (size_t i = 0; i < gxs.size(); i++)
        {
            auto x  = f->inputs[i];
            auto gx = gxs[i];

            if (!x->grad)
            {
                x->grad = gx;
            }
            else
            {
                x->grad = AsDLArrayPtr(*(x->grad) + (*gx));
            }

            if (x->creator)
            {
                AddFunc(x->creator);
            }
        }
    }

    return;
}

void Variable::ClearGrad()
{
    grad = nullptr;
}

VariablePtr Variable::Reshape(const af::dim4 shape)
{
    auto p = AsVariablePtr(*this);
    return reshape(p, shape);
}

VariablePtr Variable::Transpose()
{
    auto p = AsVariablePtr(*this);
    return transpose(p);
};

void Variable::Print(std::string desc)
{
    af::print(desc.c_str(), data);
};

// 变量转换，未来考虑去掉
VariablePtrList AsVariablePtrList(VariablePtr data)
{
    VariablePtrList l;
    l.push_back(data);
    return l;
}

NdArrayPtrList AsDLArrayPtrList(NdArray data)
{
    NdArrayPtrList l;
    l.push_back(AsDLArrayPtr(data));
    return l;
}

NdArrayPtr AsDLArrayPtr(NdArray data)
{
    return std::make_shared<NdArray>(data);
}

VariablePtr AsVariablePtr(NdArrayPtr data)
{
    return std::make_shared<Variable>(*data);
}

VariablePtr AsVariablePtr(Variable &data)
{
    return std::make_shared<Variable>(data);
}

}  // namespace dl
