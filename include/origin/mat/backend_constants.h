#ifndef __ORIGIN_DL_BACKEND_CONSTANTS_H__
#define __ORIGIN_DL_BACKEND_CONSTANTS_H__

// 预处理器宏定义，用于条件编译
#define ARRAYFIRE 1
#define EIGEN 2
#define CUSTOM 3
#define TORCH 4

namespace origin
{

// C++常量定义，用于运行时判断
constexpr int ARRAYFIRE_CONST = 1;
constexpr int EIGEN_CONST     = 2;
constexpr int CUSTOM_CONST    = 3;
constexpr int TORCH_CONST     = 4;

}  // namespace origin

#endif  // __ORIGIN_DL_BACKEND_CONSTANTS_H__
