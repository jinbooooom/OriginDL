#ifndef __ORIGIN_DL_CPU_OPS_H__
#define __ORIGIN_DL_CPU_OPS_H__

#include <memory>
#include "origin/mat/origin/../basic_types.h"
#include "origin/mat/origin/origin_mat.h"

namespace origin
{
namespace cpu
{

// === 基础运算 ===
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b);
std::unique_ptr<Mat> subtract(const OriginMat &a, const OriginMat &b);
std::unique_ptr<Mat> multiply(const OriginMat &a, const OriginMat &b);
std::unique_ptr<Mat> divide(const OriginMat &a, const OriginMat &b);
std::unique_ptr<Mat> matmul(const OriginMat &a, const OriginMat &b);

std::unique_ptr<Mat> negate(const OriginMat &mat);

// === 数学函数 ===
std::unique_ptr<Mat> exp(const OriginMat &mat);
std::unique_ptr<Mat> log(const OriginMat &mat);
std::unique_ptr<Mat> sqrt(const OriginMat &mat);
std::unique_ptr<Mat> square(const OriginMat &mat);
std::unique_ptr<Mat> pow(const OriginMat &mat, const Scalar &exponent);

// === 统计函数 ===
std::unique_ptr<Mat> sum(const OriginMat &mat, int axis);

// === 形状操作 ===
std::unique_ptr<Mat> reshape(const OriginMat &mat, const Shape &new_shape);
std::unique_ptr<Mat> transpose(const OriginMat &mat);
std::unique_ptr<Mat> broadcast_to(const OriginMat &mat, const Shape &target_shape);
std::unique_ptr<Mat> sum_to(const OriginMat &mat, const Shape &target_shape);

// === 类型转换 ===
std::unique_ptr<Mat> convert_datatype(const OriginMat &mat, DataType target_type);

}  // namespace cpu
}  // namespace origin

#endif  // __ORIGIN_DL_CPU_OPS_H__
