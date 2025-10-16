#ifndef __ORIGIN_DL_CUDA_FACTORY_H__
#define __ORIGIN_DL_CUDA_FACTORY_H__

#include "../../basic_types.h"
#include "../../../core/tensor_options.h"
#include "../../shape.h"

namespace origin
{
namespace cuda
{

// 前向声明
class OriginMat;

/**
 * @brief 在CUDA设备上创建随机张量
 * @param shape 张量形状
 * @param options 张量选项（必须指定CUDA设备）
 * @return 随机张量
 */
std::unique_ptr<OriginMat> randn(const Shape &shape, const TensorOptions &options);

/**
 * @brief 在CUDA设备上创建零张量
 * @param shape 张量形状
 * @param options 张量选项（必须指定CUDA设备）
 * @return 零张量
 */
std::unique_ptr<OriginMat> zeros(const Shape &shape, const TensorOptions &options);

/**
 * @brief 在CUDA设备上创建全1张量
 * @param shape 张量形状
 * @param options 张量选项（必须指定CUDA设备）
 * @return 全1张量
 */
std::unique_ptr<OriginMat> ones(const Shape &shape, const TensorOptions &options);

/**
 * @brief 在CUDA设备上创建填充指定值的张量
 * @param shape 张量形状
 * @param value 填充值
 * @param options 张量选项（必须指定CUDA设备）
 * @return 填充张量
 */
std::unique_ptr<OriginMat> full(const Shape &shape, double value, const TensorOptions &options);

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_FACTORY_H__
