

#include "base/dlCommon.h"

namespace dl
{

void print(const NdArray &data)
{
    af::print("", data);
}

void print(const char *str, const NdArray &data)
{
    af::print(str, data);
}

}  // namespace dl
