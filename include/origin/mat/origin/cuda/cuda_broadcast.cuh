#ifndef __ORIGIN_DL_CUDA_BROADCAST_H__
#define __ORIGIN_DL_CUDA_BROADCAST_H__

#include "../origin_mat.h"

namespace origin
{
namespace cuda
{

/**
 * @brief 计算广播形状
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 广播后的结果形状
 * @details 支持标量广播和形状匹配
 */
Shape compute_broadcast_shape(const OriginMat &a, const OriginMat &b);

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_BROADCAST_H__
