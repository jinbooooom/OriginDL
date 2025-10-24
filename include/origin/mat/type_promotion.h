#pragma once

#include "origin/core/tensor.h"
#include "origin/mat/basic_types.h"
#include <vector>

namespace origin {

/**
 * @brief 统一的类型提升工具类
 * 
 * 参考LibTorch的设计，在算子层面统一处理类型提升
 * 确保forward和backward的类型提升一致性
 */
class TypePromotion {
public:
    /**
     * @brief 检查张量列表是否需要类型提升
     * @param tensors 张量列表
     * @return true如果需要类型提升，false否则
     */
    static bool needs_promotion(const std::vector<Tensor>& tensors);
    
    /**
     * @brief 检查两个张量是否需要类型提升
     * @param a 第一个张量
     * @param b 第二个张量
     * @return true如果需要类型提升，false否则
     */
    static bool needs_promotion(const Tensor& a, const Tensor& b);
    
    /**
     * @brief 检查两个数据类型是否需要类型提升
     * @param a 第一个数据类型
     * @param b 第二个数据类型
     * @return true如果需要类型提升，false否则
     */
    static bool needs_promotion(DataType a, DataType b);
    
    /**
     * @brief 对张量列表进行类型提升
     * @param tensors 原始张量列表
     * @return 提升后的张量列表
     */
    static std::vector<Tensor> promote_tensors(const std::vector<Tensor>& tensors);
    
    /**
     * @brief 对两个张量进行类型提升
     * @param a 第一个张量
     * @param b 第二个张量
     * @return 提升后的张量对
     */
    static std::pair<Tensor, Tensor> promote_tensors(const Tensor& a, const Tensor& b);
    
    /**
     * @brief 获取两个数据类型的提升类型
     * @param a 第一个数据类型
     * @param b 第二个数据类型
     * @return 提升后的数据类型
     */
    static DataType promote_types(DataType a, DataType b);
    
    /**
     * @brief 获取张量列表的提升类型
     * @param tensors 张量列表
     * @return 提升后的数据类型
     */
    static DataType promote_types(const std::vector<Tensor>& tensors);
    
    /**
     * @brief 检查张量是否与目标类型匹配
     * @param tensor 张量
     * @param target_type 目标类型
     * @return true如果匹配，false否则
     */
    static bool is_type_match(const Tensor& tensor, DataType target_type);
    
    /**
     * @brief 将张量转换为目标类型（如果需要）
     * @param tensor 原始张量
     * @param target_type 目标类型
     * @return 转换后的张量
     */
    static Tensor to_type(const Tensor& tensor, DataType target_type);
};

/**
 * @brief 类型提升规则
 * @param a 第一个数据类型
 * @param b 第二个数据类型
 * @return 提升后的数据类型
 * @note 优先级：double > float > int64 > int32 > int16 > int8
 */
inline DataType promote_types_rule(DataType a, DataType b)
{
    // 如果类型相同，直接返回
    if (a == b)
        return a;

    // 浮点数优先级最高
    if (a == DataType::kFloat64 || b == DataType::kFloat64)
        return DataType::kFloat64;
    if (a == DataType::kFloat32 || b == DataType::kFloat32)
        return DataType::kFloat32;

    // 整数类型按精度排序
    if (a == DataType::kInt64 || b == DataType::kInt64)
        return DataType::kInt64;
    if (a == DataType::kInt32 || b == DataType::kInt32)
        return DataType::kInt32;
    if (a == DataType::kInt16 || b == DataType::kInt16)
        return DataType::kInt16;
    if (a == DataType::kInt8 || b == DataType::kInt8)
        return DataType::kInt8;

    // 无符号整数
    if (a == DataType::kUInt64 || b == DataType::kUInt64)
        return DataType::kUInt64;
    if (a == DataType::kUInt32 || b == DataType::kUInt32)
        return DataType::kUInt32;
    if (a == DataType::kUInt16 || b == DataType::kUInt16)
        return DataType::kUInt16;
    if (a == DataType::kUInt8 || b == DataType::kUInt8)
        return DataType::kUInt8;

    // 布尔类型
    if (a == DataType::kBool || b == DataType::kBool)
        return DataType::kBool;

    // 默认返回第一个类型
    return a;
}

} // namespace origin
