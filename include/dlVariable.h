#ifndef __DLZERO_VARIABLE_H__
#define __DLZERO_VARIABLE_H__

#include "base/dlDefine.h"

namespace dl
{
class Variable
{
  public:
    NdArray data;

    Variable(const NdArray &data);

    virtual ~Variable();

    void Print();
};

}  // namespace dl

#endif