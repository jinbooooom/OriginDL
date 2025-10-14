#include "origin/mat/origin/origin_mat.h"
#include <algorithm>
#include <stdexcept>

namespace origin {
namespace cpu {

std::unique_ptr<OriginMat> sum_to(const OriginMat& mat, const Shape& target_shape) {
    // 检查形状兼容性
    if (mat.shape() == target_shape) {
        // 形状相同，直接返回副本
        return std::make_unique<OriginMat>(mat);
    }
    
    // 计算元素总数
    size_t current_elements = mat.elements();
    size_t target_elements = target_shape.elements();
    
    if (target_elements > current_elements) {
        // 目标形状更大，sum_to不支持广播，抛出异常
        throw std::runtime_error("sum_to: Target shape cannot have more elements than source tensor");
    } else {
        // 目标形状更小或相等，需要求和压缩
        // 收集需要求和的维度
        std::vector<int> sum_dims;
        
        // 从左到右比较维度（按照torch_mat的逻辑）
        size_t min_dims = std::min(mat.shape().size(), target_shape.size());
        for (size_t i = 0; i < min_dims; ++i) {
            if (target_shape[i] == 1 && mat.shape()[i] > 1) {
                sum_dims.push_back(i);
            }
        }
        
        // 处理多余的维度（从右边开始的多余维度）
        // 如果源形状比目标形状多维度，需要对这些维度求和
        if (mat.shape().size() > target_shape.size()) {
            for (size_t i = target_shape.size(); i < mat.shape().size(); ++i) {
                sum_dims.push_back(i);
            }
        }
        
        // 执行求和操作
        std::unique_ptr<OriginMat> current = std::make_unique<OriginMat>(mat);
        
        // 按从大到小的顺序求和，这样轴索引不会改变
        std::sort(sum_dims.begin(), sum_dims.end(), std::greater<int>());
        for (int dim : sum_dims) {
            auto sum_result = current->sum(dim);
            current = std::unique_ptr<OriginMat>(static_cast<OriginMat*>(sum_result.release()));
        }
        
        // 最后reshape到目标形状
        if (current->shape() != target_shape) {
            auto reshape_result = current->reshape(target_shape);
            current = std::unique_ptr<OriginMat>(static_cast<OriginMat*>(reshape_result.release()));
        }
        
        return current;
    }
}

} // namespace cpu
} // namespace origin
