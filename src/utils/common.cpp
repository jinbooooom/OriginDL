

#include "origin/mat/backend.h"

namespace origin
{

void print(const NdArray &data)
{
    // #if MAT_BACKEND == ARRAYFIRE
    //     af::print("", data);
    // #elif MAT_BACKEND == TORCH
    std::cout << data << std::endl;
    // #endif
}

void print(const char *str, const NdArray &data)
{
    // #if MAT_BACKEND == ARRAYFIRE
    //     af::print(str, data);
    // #elif MAT_BACKEND == TORCH
    std::cout << str << std::endl << data << std::endl;
    // #endif
}

}  // namespace origin
