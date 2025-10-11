#ifndef __ORIGIN_DL_INNER_TYPES_H__
#define __ORIGIN_DL_INNER_TYPES_H__

#include <memory>

namespace origin
{

// 前向声明
class Operator;
using FunctionPtr = std::shared_ptr<Operator>;

}  // namespace origin

#endif