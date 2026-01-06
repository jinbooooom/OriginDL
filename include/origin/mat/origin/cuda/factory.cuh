#ifndef __ORIGIN_DL_CUDA_FACTORY_H__
#define __ORIGIN_DL_CUDA_FACTORY_H__

#include "../../../core/tensor_options.h"
#include "../../basic_types.h"
#include "../../shape.h"

namespace origin
{
namespace cuda
{

// 前向声明
class OriginMat;

/**
 * @brief 在CUDA设备上创建零张量
 * @param shape 张量形状
 * @param options 张量选项（必须指定CUDA设备）
 * @return 零张量
 */
std::unique_ptr<origin::OriginMat> zeros(const Shape &shape, const TensorOptions &options);

/**
 * @brief 在CUDA设备上创建全1张量
 * @param shape 张量形状
 * @param options 张量选项（必须指定CUDA设备）
 * @return 全1张量
 */
std::unique_ptr<origin::OriginMat> ones(const Shape &shape, const TensorOptions &options);

/**
 * @brief 在CUDA设备上创建填充指定值的张量
 * @param shape 张量形状
 * @param scalar 填充值
 * @param options 张量选项（必须指定CUDA设备）
 * @return 填充张量
 */
std::unique_ptr<origin::OriginMat> full(const Shape &shape, const Scalar &scalar, const TensorOptions &options);

/**
 * @brief 从内存数据创建CUDA张量
 * @param data 原始数据指针
 * @param user_dtype 用户数据类型
 * @param shape 张量形状
 * @param options 张量选项（必须指定CUDA设备）
 * @return 创建的张量
 */
std::unique_ptr<origin::OriginMat> from_memory(const void *data,
                                               DataType user_dtype,
                                               const Shape &shape,
                                               const TensorOptions &options);

/**
 * @brief 在CUDA设备上拼接多个张量
 * @param inputs 输入矩阵列表
 * @param dim 拼接维度
 * @return 拼接结果矩阵
 */
std::unique_ptr<origin::Mat> cat(const std::vector<const origin::OriginMat *> &inputs, int dim);

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_FACTORY_H__
