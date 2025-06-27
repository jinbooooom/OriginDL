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
// #include "spdlog.h"
#include "dlLog.h"

namespace dl
{

using data_t  = double;
using NdArray = af::array;

class Variable;
class Function;
using NdArrayPtr  = std::shared_ptr<NdArray>;
using FunctionPtr = std::shared_ptr<Function>;
using VariablePtr = std::shared_ptr<Variable>;

extern void print(const NdArray &data);

}  // namespace dl

#endif
