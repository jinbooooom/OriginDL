#include "origin/mat/origin/origin_mat.h"
#include <stdexcept>

namespace origin {
namespace cpu {

// 前向声明
data_t sum_all(const OriginMat& mat);

data_t mean_all(const OriginMat& mat) {
    return sum_all(mat) / static_cast<data_t>(mat.elements());
}

} // namespace cpu
} // namespace origin
