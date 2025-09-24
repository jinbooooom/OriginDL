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

class Operator;
using NdArrayPtr       = std::shared_ptr<NdArray>;
using NdArrayPtrList   = std::vector<NdArrayPtr>;
using NdArrayList      = std::vector<NdArray>;
using FunctionPtr      = std::shared_ptr<Operator>;

extern void print(const NdArray &data);

}  // namespace dl

#endif
