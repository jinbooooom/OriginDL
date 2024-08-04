#ifndef __DLZERO_DEFINE_H__
#define __DLZERO_DEFINE_H__

#include <cmath>
#include <functional>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "NumCpp.hpp"

namespace dl
{

using data_t  = double;
using NdArray = nc::NdArray<data_t>;

static void print(const NdArray &data)
{
    std::cout << data << std::endl;
}

}  // namespace dl

#endif