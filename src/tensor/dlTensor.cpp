#include "base/dlException.h"
#include "originDL.h"

namespace dl
{
Variable::Variable(const NdArray &data) : data_(data), generation_(0) {}

Variable::~Variable() {}

void Variable::SetCreator(const FunctionPtr &func)
{
    creator_    = func;
    generation_ = creator_->generation_ + 1;
}

void Variable::Backward()
{
    if (!this->grad_)
    {
        double grad_val = 1.0;
        auto dims       = this->data_.dims();
        grad_           = std::make_shared<NdArray>(af::constant(grad_val, dims));
    }

    auto funcs   = std::list<FunctionPtr>();
    auto funcSet = std::set<FunctionPtr>();  // 考虑到多输出的情况下

    auto AddFunc = [&funcs, &funcSet](const FunctionPtr &f) {
        if (funcSet.find(f) == funcSet.end())
        {
            funcs.push_back(f);
            funcSet.insert(f);
            funcs.sort(
                [](const FunctionPtr &lhs, const FunctionPtr &rhs) { return lhs->generation_ < rhs->generation_; });
        }
    };

    AddFunc(this->creator_);

    while (!funcs.empty())
    {
        auto f = funcs.back();
        funcs.pop_back();

        auto gys = NdArrayPtrList();
        for (const auto &o : f->outputs_)
        {
            // 通过 lock() 升级为 shared_ptr 并检查有效性
            if (auto oPtr = o.lock())
            {
                gys.emplace_back(oPtr->grad_);
            }
            else
            {
                DL_CRITICAL_THROW("backward error!, output is nullptr");
            }
        }
        auto gxs = f->Backward(gys);

        if (gxs.size() != f->inputs_.size())
        {
            DL_ERROR_THROW("backward error!, gxs size " + std::to_string(gxs.size()) + ", inputs size " +
                           std::to_string(f->inputs_.size()));
        }

        for (size_t i = 0; i < gxs.size(); i++)
        {
            auto x  = f->inputs_[i];
            auto gx = gxs[i];

            if (!x->grad_)
            {
                x->grad_ = gx;
            }
            else
            {
                x->grad_ = AsDLArrayPtr(*(x->grad_) + (*gx));
            }

            if (x->creator_)
            {
                AddFunc(x->creator_);
            }
        }
    }

    return;
}

void Variable::ClearGrad()
{
    grad_ = nullptr;
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
    af::print(desc.c_str(), data_);
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
