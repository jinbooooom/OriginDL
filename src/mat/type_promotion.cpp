#include "origin/mat/type_promotion.h"
#include "origin/mat/basic_types.h"
#include <algorithm>

namespace origin {

bool TypePromotion::needs_promotion(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        return false;
    }
    
    DataType first_type = tensors[0].dtype();
    return std::any_of(tensors.begin(), tensors.end(), 
                      [first_type](const Tensor& t) { return t.dtype() != first_type; });
}

bool TypePromotion::needs_promotion(const Tensor& a, const Tensor& b) {
    return a.dtype() != b.dtype();
}

bool TypePromotion::needs_promotion(DataType a, DataType b) {
    return a != b;
}

std::vector<Tensor> TypePromotion::promote_tensors(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        return tensors;
    }
    
    // 获取提升后的类型
    DataType promoted_type = promote_types(tensors);
    
    // 转换所有张量到提升后的类型
    std::vector<Tensor> promoted_tensors;
    promoted_tensors.reserve(tensors.size());
    
    for (const auto& tensor : tensors) {
        promoted_tensors.push_back(to_type(tensor, promoted_type));
    }
    
    return promoted_tensors;
}

std::pair<Tensor, Tensor> TypePromotion::promote_tensors(const Tensor& a, const Tensor& b) {
    if (!needs_promotion(a, b)) {
        return {a, b};
    }
    
    DataType promoted_type = promote_types(a.dtype(), b.dtype());
    return {to_type(a, promoted_type), to_type(b, promoted_type)};
}

DataType TypePromotion::promote_types(DataType a, DataType b) {
    // 使用类型提升规则函数
    return ::origin::promote_types_rule(a, b);
}

DataType TypePromotion::promote_types(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        return DataType::kFloat32; // 默认类型
    }
    
    DataType result = tensors[0].dtype();
    for (size_t i = 1; i < tensors.size(); ++i) {
        result = promote_types(result, tensors[i].dtype());
    }
    
    return result;
}

bool TypePromotion::is_type_match(const Tensor& tensor, DataType target_type) {
    return tensor.dtype() == target_type;
}

Tensor TypePromotion::to_type(const Tensor& tensor, DataType target_type) {
    if (is_type_match(tensor, target_type)) {
        return tensor;
    }
    return tensor.to(target_type);
}

} // namespace origin
