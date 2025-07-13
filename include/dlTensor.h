#ifndef __ORIGIN_DL_VARIABLE_H__
#define __ORIGIN_DL_VARIABLE_H__

#include "base/dlCommon.h"

namespace dl
{
class Variable
{
  public:
    NdArray data;

    NdArrayPtr grad;

    FunctionPtr creator;

    int generation;

    Variable(const NdArray &data);

    virtual ~Variable();

    void SetCreator(const FunctionPtr &func);

    void Backward();

    void ClearGrad();

    // 矩阵方法
    VariablePtr Reshape(const af::dim4 shape);

    VariablePtr Transpose();

    // 调试
    void Print();
};

extern VariablePtrList AsVariablePtrList(VariablePtr data);

extern NdArrayPtrList AsDLArrayPtrList(NdArray data);

extern NdArrayPtr AsDLArrayPtr(NdArray data);

extern VariablePtr AsVariablePtr(NdArrayPtr data);

extern VariablePtr AsVariablePtr(Variable &data);

}  // namespace dl

#endif
