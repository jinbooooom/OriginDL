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
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b);

/**
 * @brief CUDA减法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 减法结果矩阵
 */
std::unique_ptr<Mat> subtract(const OriginMat &a, const OriginMat &b);

/**
 * @brief CUDA乘法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 乘法结果矩阵
 */
std::unique_ptr<Mat> multiply(const OriginMat &a, const OriginMat &b);

/**
 * @brief CUDA除法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 除法结果矩阵
 */
std::unique_ptr<Mat> divide(const OriginMat &a, const OriginMat &b);

// ============================================================================
// 一元运算算子
// ============================================================================

/**
 * @brief CUDA指数算子
 * @param mat 输入矩阵
 * @return 指数运算结果矩阵
 */
std::unique_ptr<Mat> exp(const OriginMat &mat);

/**
 * @brief CUDA对数算子
 * @param mat 输入矩阵
 * @return 对数运算结果矩阵
 */
std::unique_ptr<Mat> log(const OriginMat &mat);

/**
 * @brief CUDA平方根算子
 * @param mat 输入矩阵
 * @return 平方根运算结果矩阵
 */
std::unique_ptr<Mat> sqrt(const OriginMat &mat);

/**
 * @brief CUDA平方算子
 * @param mat 输入矩阵
 * @return 平方运算结果矩阵
 */
std::unique_ptr<Mat> square(const OriginMat &mat);

/**
 * @brief CUDA取负算子
 * @param mat 输入矩阵
 * @return 取负运算结果矩阵
 */
std::unique_ptr<Mat> negate(const OriginMat &mat);

// ============================================================================
// 标量运算算子
// ============================================================================

/**
 * @brief CUDA标量加法算子
 * @param mat 输入矩阵
 * @param scalar 标量值
 * @return 标量加法结果矩阵
 */
std::unique_ptr<Mat> add_scalar(const OriginMat &mat, data_t scalar);

/**
 * @brief CUDA标量乘法算子
 * @param mat 输入矩阵
 * @param scalar 标量值
 * @return 标量乘法结果矩阵
 */
std::unique_ptr<Mat> multiply_scalar(const OriginMat &mat, data_t scalar);

// ============================================================================
// 高级运算算子（待实现）
// ============================================================================

/**
 * @brief CUDA矩阵乘法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 矩阵乘法结果矩阵
 * @note 待实现
 */
// std::unique_ptr<Mat> matmul(const OriginMat& a, const OriginMat& b);

/**
 * @brief CUDA求和算子
 * @param mat 输入矩阵
 * @param axis 求和轴，-1表示所有元素求和
 * @return 求和结果矩阵
 * @note 待实现
 */
// std::unique_ptr<Mat> sum(const OriginMat& mat, int axis = -1);

/**
 * @brief CUDA转置算子
 * @param mat 输入矩阵
 * @return 转置结果矩阵
 * @note 待实现
 */
// std::unique_ptr<Mat> transpose(const OriginMat& mat);

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_OPS_H__
