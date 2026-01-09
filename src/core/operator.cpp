#include "origin/core/operator.h"

namespace origin
{

// 定义静态成员变量：用于标记空 Tensor，区分一元和二元原地操作
const Tensor Operator::kNullTensor_{};

}  // namespace origin

