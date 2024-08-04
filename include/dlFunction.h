#ifndef __DLZERO_FUNCTION_H__
#define __DLZERO_FUNCTION_H__

#include "dlVariable.h"

namespace dl
{

class Function
{
  public:
    virtual ~Function() {}

    Variable operator()(const Variable &input);

    virtual NdArray forward(const NdArray &x) = 0;
};

class Square : public Function
{
  public:
    NdArray forward(const NdArray &x) override;
};

}  // namespace dl

#endif