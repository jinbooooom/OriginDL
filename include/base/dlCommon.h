#ifndef __DLZERO_COMMON_H__
#define __DLZERO_COMMON_H__

#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "NumCpp.hpp"
// #include "spdlog.h"
#include "dlLog.h"

namespace dl
{

using data_t  = double;
using NdArray = nc::NdArray<data_t>;

extern void print(const NdArray &data);

}  // namespace dl

#endif