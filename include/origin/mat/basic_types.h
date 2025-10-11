#ifndef __ORIGIN_DL_BASIC_TYPES_H__
#define __ORIGIN_DL_BASIC_TYPES_H__

namespace origin
{

// 基础数据类型定义
using data_t = float;

// 矩阵计算后端的类型
constexpr int ORIGIN_BACKEND_TYPE = 0;
constexpr int TORCH_BACKEND_TYPE  = 1;

}  // namespace origin

#endif  // __ORIGIN_DL_BASIC_TYPES_H__
