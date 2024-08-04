#ifndef __DLZERO_VARIABLE_H__
#define __DLZERO_VARIABLE_H__

#include "base/dlCommon.h"

namespace dl
{
class Variable
{
  public:
    NdArray data;
    std::shared_ptr<NdArray> grad;

    Variable(const NdArray &data);

    virtual ~Variable();

    void Print();
};

}  // namespace dl

#endif