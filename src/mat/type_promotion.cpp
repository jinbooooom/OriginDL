#include "origin/mat/type_promotion.h"
#include <algorithm>
#include "origin/mat/basic_types.h"
#include "origin/utils/branch_prediction.h"

namespace origin
{

bool TypePromotion::needs_promotion(const std::vector<Tensor> &tensors)
{
    if (unlikely(tensors.empty()))
    {
        return false;
    }

    DataType first_type = tensors[0].dtype();
    return std::any_of(tensors.begin(), tensors.end(),
                       [first_type](const Tensor &t) { return t.dtype() != first_type; });
}

std::vector<Tensor> TypePromotion::promote_tensors(const std::vector<Tensor> &tensors)
{
    if (unlikely(tensors.empty()))
    {
        return tensors;
    }

    // 获取提升后的类型
    DataType promoted_type = promote_types(tensors);

    // 转换所有张量到提升后的类型
    std::vector<Tensor> promoted_tensors;
    promoted_tensors.reserve(tensors.size());

    for (const auto &tensor : tensors)
    {
        promoted_tensors.push_back(to_type(tensor, promoted_type));
    }

    return promoted_tensors;
}

std::pair<Tensor, Tensor> TypePromotion::promote_tensors(const Tensor &a, const Tensor &b)
{
    if (!needs_promotion(a, b))
    {
        return {a, b};
    }

    DataType promoted_type = promote_types(a.dtype(), b.dtype());
    return {to_type(a, promoted_type), to_type(b, promoted_type)};
}

DataType TypePromotion::promote_types(const std::vector<Tensor> &tensors)
{
    if (unlikely(tensors.empty()))
    {
        return DataType::kFloat32;  // 默认类型
    }

    DataType result = tensors[0].dtype();
    for (size_t i = 1; i < tensors.size(); ++i)
    {
        result = promote_types(result, tensors[i].dtype());
    }

    return result;
}

std::pair<MaybeOwned<Tensor>, MaybeOwned<Tensor>> TypePromotion::promote_tensors_maybe_owned(const Tensor &a, const Tensor &b)
{
    if (!needs_promotion(a, b))
    {
        // 类型相同，直接借用，零开销
        return {MaybeOwned<Tensor>::borrowed(a), MaybeOwned<Tensor>::borrowed(b)};
    }

    DataType promoted_type = promote_types(a.dtype(), b.dtype());
    
    // 使用 MaybeOwned 优化：类型匹配时借用，不匹配时拥有
    return {
        to_type_maybe_owned(a, promoted_type),
        to_type_maybe_owned(b, promoted_type)
    };
}

Tensor TypePromotion::to_type(const Tensor &tensor, DataType target_type)
{
    if (is_type_match(tensor, target_type))
    {
        return tensor;
    }
    return tensor.to(target_type);
}

MaybeOwned<Tensor> TypePromotion::to_type_maybe_owned(const Tensor &tensor, DataType target_type)
{
    if (is_type_match(tensor, target_type))
    {
        // 类型匹配，借用引用，零开销（不增加 shared_ptr 引用计数）
        return MaybeOwned<Tensor>::borrowed(tensor);
    }
    // 类型不匹配，创建新对象并拥有所有权
    return MaybeOwned<Tensor>::owned(tensor.to(target_type));
}

}  // namespace origin
