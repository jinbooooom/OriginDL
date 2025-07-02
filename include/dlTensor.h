#ifndef __ORIGIN_DL_VARIABLE_H__
#define __ORIGIN_DL_VARIABLE_H__

#include "base/dlCommon.h"
#include "dlFunction.h"

namespace dl
{
class Variable
{
  public:
    NdArray data;

    NdArrayPtr grad;

    FunctionPtr creator;

    Variable(const NdArray &data);

    virtual ~Variable();

    void SetCreator(const FunctionPtr &func);

    void Backward();

    void Print();
};

extern VariablePtrList AsVariablePtrList(VariablePtr data);

// extern NdArrayPtrList AsNdArrayPtrList(VariablePtr data);

extern NdArrayPtr AsDLArrayPtr(NdArray data);

extern VariablePtr AsVariablePtr(NdArrayPtr data);

extern VariablePtr AsVariablePtr(Variable &data);

}  // namespace dl

#endif