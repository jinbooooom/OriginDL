#ifndef __ORIGIN_DL_CUDA_OPS_H__
#define __ORIGIN_DL_CUDA_OPS_H__

#include <memory>
#include "../origin_mat.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA运算接口声明
 * @details 提供所有CUDA算子的统一接口
 */

// ============================================================================
// 基础二元运算算子
// ============================================================================

/**
 * @brief CUDA加法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 加法结果矩阵
 */
std::unique_ptr<Mat> add(const origin::OriginMat &a, const origin::OriginMat &b);

/**
 * @brief CUDA减法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 减法结果矩阵
 */
std::unique_ptr<Mat> subtract(const origin::OriginMat &a, const origin::OriginMat &b);

/**
 * @brief CUDA乘法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 乘法结果矩阵
 */
std::unique_ptr<Mat> multiply(const origin::OriginMat &a, const origin::OriginMat &b);

/**
 * @brief CUDA除法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 除法结果矩阵
 */
std::unique_ptr<Mat> divide(const origin::OriginMat &a, const origin::OriginMat &b);

// ============================================================================
// 一元运算算子
// ============================================================================

/**
 * @brief CUDA指数算子
 * @param mat 输入矩阵
 * @return 指数运算结果矩阵
 */
std::unique_ptr<Mat> exp(const origin::OriginMat &mat);

/**
 * @brief CUDA对数算子
 * @param mat 输入矩阵
 * @return 对数运算结果矩阵
 */
std::unique_ptr<Mat> log(const origin::OriginMat &mat);

/**
 * @brief CUDA平方根算子
 * @param mat 输入矩阵
 * @return 平方根运算结果矩阵
 */
std::unique_ptr<Mat> sqrt(const origin::OriginMat &mat);

/**
 * @brief CUDA平方算子
 * @param mat 输入矩阵
 * @return 平方运算结果矩阵
 */
std::unique_ptr<Mat> square(const origin::OriginMat &mat);

/**
 * @brief CUDA取负算子
 * @param mat 输入矩阵
 * @return 取负运算结果矩阵
 */
std::unique_ptr<Mat> negate(const origin::OriginMat &mat);

// ============================================================================
// 标量运算算子
// ============================================================================

/**
 * @brief CUDA标量加法算子
 * @param mat 输入矩阵
 * @param scalar 标量值
 * @return 标量加法结果矩阵
 */
std::unique_ptr<Mat> add_scalar(const origin::OriginMat &mat, data_t scalar);

/**
 * @brief CUDA标量乘法算子
 * @param mat 输入矩阵
 * @param scalar 标量值
 * @return 标量乘法结果矩阵
 */
std::unique_ptr<Mat> multiply_scalar(const origin::OriginMat &mat, data_t scalar);

// ============================================================================
// 形状操作算子
// ============================================================================

/**
 * @brief CUDA重塑算子
 * @param mat 输入矩阵
 * @param new_shape 新的形状
 * @return 重塑后的矩阵
 */
std::unique_ptr<Mat> reshape(const origin::OriginMat &mat, const Shape &new_shape);

/**
 * @brief CUDA转置算子
 * @param mat 输入矩阵
 * @return 转置后的矩阵
 */
std::unique_ptr<Mat> transpose(const origin::OriginMat &mat);

// ============================================================================
// 高级运算算子
// ============================================================================

/**
 * @brief CUDA矩阵乘法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 矩阵乘法结果矩阵
 */
std::unique_ptr<Mat> matmul(const OriginMat &a, const OriginMat &b);

/**
 * @brief CUDA求和算子
 * @param mat 输入矩阵
 * @param axis 求和轴，-1表示所有元素求和
 * @return 求和结果矩阵
 */
std::unique_ptr<Mat> sum(const OriginMat &mat, int axis = -1);

/**
 * @brief CUDA幂运算算子（标量）
 * @param mat 输入矩阵
 * @param exponent 指数
 * @return 幂运算结果矩阵
 */
std::unique_ptr<Mat> pow(const OriginMat &mat, data_t exponent);

/**
 * @brief CUDA幂运算算子（张量）
 * @param base 底数矩阵
 * @param exponent 指数矩阵
 * @return 幂运算结果矩阵
 */
std::unique_ptr<Mat> pow(const OriginMat &base, const OriginMat &exponent);

/**
 * @brief CUDA广播算子
 * @param mat 输入矩阵
 * @param target_shape 目标形状
 * @return 广播后的矩阵
 */
std::unique_ptr<Mat> broadcast_to(const OriginMat &mat, const Shape &target_shape);

/**
 * @brief CUDA sum_to算子
 * @param mat 输入矩阵
 * @param target_shape 目标形状
 * @return sum_to结果矩阵
 */
std::unique_ptr<Mat> sum_to(const OriginMat &mat, const Shape &target_shape);

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_OPS_H__
