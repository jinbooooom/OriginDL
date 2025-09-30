#ifndef __ORIGIN_DL_TYPES_H__
#define __ORIGIN_DL_TYPES_H__

#include <memory>
#include <vector>
#include "../mat/backend.h"
#include "../mat/basic_types.h"

namespace origin
{

// 前向声明
class Operator;

// 类型别名
using NdArrayPtr     = std::shared_ptr<NdArray>;
using NdArrayPtrList = std::vector<NdArrayPtr>;
using NdArrayList    = std::vector<NdArray>;
using FunctionPtr    = std::shared_ptr<Operator>;

}  // namespace origin

#endif  // __ORIGIN_DL_TYPES_H__
