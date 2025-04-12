#ifndef __DLZERO_VARIABLE_H__
#define __DLZERO_VARIABLE_H__

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

    void SetCreator(const FunctionPtr& func);

	  void Backward();

    void Print();
};

}  // namespace dl



#endif