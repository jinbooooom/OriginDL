#ifndef __ORIGIN_DL_VARIABLE_H__
#define __ORIGIN_DL_VARIABLE_H__

#include "base/dlCommon.h"

namespace dl
{
class Variable
{
  public:
    NdArray data_;

    NdArrayPtr grad_;

    FunctionPtr creator_;

    int generation_;

    Variable(const NdArray &data);

    virtual ~Variable();

    void set_creator(const FunctionPtr &func);

    void backward();

    void clear_grad();

    // 矩阵方法
    VariablePtr reshape(const af::dim4 shape);

    VariablePtr transpose();

    // 调试
    void print(std::string desc = "");
};

extern VariablePtrList as_variable_ptr_list(VariablePtr data);

extern NdArrayPtrList as_dl_array_ptr_list(NdArray data);

extern NdArrayPtr as_dl_array_ptr(NdArray data);

extern VariablePtr as_variable_ptr(NdArrayPtr data);

extern VariablePtr as_variable_ptr(Variable &data);

}  // namespace dl

#endif
