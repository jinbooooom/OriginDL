#ifndef __ORIGIN_DL_UTILS_H__
#define __ORIGIN_DL_UTILS_H__

#include "dlCommon.h"

namespace dl
{
namespace utils
{
af::array BroadcastTo(const af::array &src, const af::dim4 &targetShape);

af::array SumTo(const af::array &src, const af::dim4 &targetShape);
}  // namespace utils
}  // namespace dl

#endif