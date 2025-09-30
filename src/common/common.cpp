

#include "base/common.h"

namespace origin
{

void print(const NdArray &data)
{
    af::print("", data);
}

void print(const char *str, const NdArray &data)
{
    af::print(str, data);
}

}  // namespace origin
