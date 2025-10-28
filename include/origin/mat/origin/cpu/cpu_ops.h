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
std::unique_ptr<OriginMat> add(const OriginMat &a, const OriginMat &b);
std::unique_ptr<OriginMat> subtract(const OriginMat &a, const OriginMat &b);
std::unique_ptr<OriginMat> multiply(const OriginMat &a, const OriginMat &b);
std::unique_ptr<OriginMat> divide(const OriginMat &a, const OriginMat &b);
std::unique_ptr<OriginMat> matmul(const OriginMat &a, const OriginMat &b);

std::unique_ptr<OriginMat> negate(const OriginMat &mat);

// === 数学函数 ===
std::unique_ptr<OriginMat> exp(const OriginMat &mat);
std::unique_ptr<OriginMat> log(const OriginMat &mat);
std::unique_ptr<OriginMat> sqrt(const OriginMat &mat);
std::unique_ptr<OriginMat> square(const OriginMat &mat);
std::unique_ptr<OriginMat> pow(const OriginMat &mat, const Scalar &exponent);

// === 统计函数 ===
std::unique_ptr<OriginMat> sum(const OriginMat &mat, int axis);
data_t sum_all(const OriginMat &mat);
data_t max_all(const OriginMat &mat);
data_t min_all(const OriginMat &mat);
data_t mean_all(const OriginMat &mat);

// === 形状操作 ===
std::unique_ptr<OriginMat> reshape(const OriginMat &mat, const Shape &new_shape);
std::unique_ptr<OriginMat> transpose(const OriginMat &mat);
std::unique_ptr<OriginMat> broadcast_to(const OriginMat &mat, const Shape &target_shape);
std::unique_ptr<OriginMat> sum_to(const OriginMat &mat, const Shape &target_shape);

// === 类型转换 ===
std::unique_ptr<OriginMat> convert_datatype(const OriginMat &mat, DataType target_type);

// === 工厂方法 ===
std::unique_ptr<OriginMat> randn(const Shape &shape, const TensorOptions &options);
std::unique_ptr<OriginMat> zeros(const Shape &shape, const TensorOptions &options);
std::unique_ptr<OriginMat> ones(const Shape &shape, const TensorOptions &options);
std::unique_ptr<OriginMat> full(const Shape &shape, const Scalar &scalar, const TensorOptions &options);
std::unique_ptr<OriginMat> from_memory(const void *data, DataType user_dtype, const Shape &shape, const TensorOptions &options);

}  // namespace cpu
}  // namespace origin

#endif  // __ORIGIN_DL_CPU_OPS_H__
