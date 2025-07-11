#ifndef __ORIGIN_DL_COMMON_H__
#define __ORIGIN_DL_COMMON_H__

#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <arrayfire.h>
#include "dlLog.h"

namespace dl
{

using data_t  = double;
using NdArray = af::array;
using DLMat   = af::array;

class Variable;
class Operator;
using NdArrayPtr       = std::shared_ptr<NdArray>;
using NdArrayPtrList   = std::vector<NdArrayPtr>;
using FunctionPtr      = std::shared_ptr<Operator>;
using VariablePtr      = std::shared_ptr<Variable>;
using VariablePtrList  = std::vector<VariablePtr>;
using VariableWPtr     = std::weak_ptr<Variable>;
using VariableWPtrList = std::vector<VariableWPtr>;

extern void print(const NdArray &data);

}  // namespace dl

#endif
