#ifndef __ORIGIN_DL_CUDA_OPS_H__
#define __ORIGIN_DL_CUDA_OPS_H__

#include <memory>
#include "../origin_mat.h"

namespace origin
{
namespace cuda
{

// CUDA运算接口声明
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b);

// 未来扩展其他算子
// std::unique_ptr<Mat> subtract(const OriginMat& a, const OriginMat& b);
// std::unique_ptr<Mat> multiply(const OriginMat& a, const OriginMat& b);
// std::unique_ptr<Mat> divide(const OriginMat& a, const OriginMat& b);

}  // namespace cuda
}  // namespace origin

#endif
