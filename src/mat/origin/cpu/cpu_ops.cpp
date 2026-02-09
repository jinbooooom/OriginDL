/**
 * @file cpu_ops.cpp
 * @brief CPU 非计算类算子实现
 * 
 * ============================================================================
 * 文件功能说明
 * ============================================================================
 * 
 * 本文件承担非计算类 CPU 算子的实现，类似于 add.cpp 但按功能分类而非按算子分类。
 * 
 * 架构位置：
 * - origin_mat.cpp (封装层)
 *   ↓ 包含
 * - cpu_ops.h (所有 CPU 算子的接口声明)
 *   ↓ 声明
 * - cpu_ops.cpp (本文件：非计算类算子实现：clone、index_put)
 * - add.cpp, divide.cpp 等 (计算类算子实现)
 *   ↓ 都包含
 * - cpu_kernels.h (基础操作定义，只在 .cpp 文件中使用)
 * 
 * ============================================================================
 */

#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"
#include <cstring>

namespace origin
{
namespace cpu
{

// ============================================================================
// 非计算类算子实现
// ============================================================================

/**
 * @brief CPU clone：深拷贝张量（支持非连续张量）
 * @param mat 输入矩阵
 * @return 拷贝后的矩阵（连续存储）
 */
std::unique_ptr<Mat> clone(const OriginMat &mat)
{
    // 深拷贝：创建新的 Storage 并复制数据（真正的独立副本）
    size_t data_size = mat.elements() * element_size(mat.dtype());
    auto new_storage = Storage::create(data_size, mat.device().type(), mat.device().index());

    // 如果张量是连续的，可以直接使用 memcpy（快速路径）
    if (mat.is_contiguous())
    {
        std::memcpy(new_storage->data(), mat.storage()->data(), data_size);
    }
    else
    {
        // 对于非连续张量，需要按逻辑顺序拷贝（使用 strides）
        // 计算目标张量的连续 strides（用于写入和构造 OriginMat）
        auto output_strides = utils::compute_strides(mat.shape());

        // CPU 版本：按逻辑顺序逐个拷贝元素
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            const T *src_data = static_cast<const T *>(mat.storage()->data());
            T *dst_data       = static_cast<T *>(new_storage->data());

            size_t total_elements = mat.elements();
            size_t ndim           = mat.shape().size();

            // 遍历所有逻辑索引
            for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx)
            {
                // 将线性索引转换为多维坐标（按输出形状）
                std::vector<size_t> coords(ndim);
                size_t remaining = linear_idx;
                for (size_t d = 0; d < ndim; ++d)
                {
                    coords[d] = remaining / output_strides[d];
                    remaining %= output_strides[d];
                }

                // 计算源张量的物理偏移（使用实际的 strides）
                size_t src_offset = 0;
                for (size_t d = 0; d < ndim; ++d)
                {
                    src_offset += coords[d] * mat.strides()[d];
                }

                // 拷贝元素（目标位置是连续的，所以直接使用 linear_idx）
                dst_data[linear_idx] = src_data[src_offset];
            }
        });

        // 使用已计算的 output_strides 创建 OriginMat，避免构造函数中重复计算
        return std::make_unique<OriginMat>(new_storage, mat.shape(), output_strides, mat.dtype());
    }

    // 创建新的 OriginMat，使用新的 Storage（连续情况，构造函数会计算 strides）
    return std::make_unique<OriginMat>(new_storage, mat.shape(), mat.dtype());
}

/**
 * @brief CPU index_put：根据多维索引写入单个元素
 * @param mat 输入/输出矩阵（原地修改）
 * @param indices 多维索引
 * @param value 要写入的标量值
 */
void index_put(OriginMat &mat, std::initializer_list<size_t> indices, const Scalar &value)
{
    if (unlikely(indices.size() != mat.shape().size()))
    {
        THROW_INVALID_ARG("Index count ({}) does not match tensor dimension ({}). Indices: {}, Shape: {}",
                          indices.size(), mat.shape().size(), "[indices]", mat.shape().to_string());
    }

    // 验证每个索引值并计算内存偏移（使用 strides，支持非连续内存）
    size_t offset = 0;
    size_t i      = 0;
    for (auto idx : indices)
    {
        if (unlikely(idx >= mat.shape()[i]))
        {
            THROW_INVALID_ARG("Index {} out of range for dimension {} (size: {}). Indices: {}, Shape: {}", idx, i,
                              mat.shape()[i], "[indices]", mat.shape().to_string());
        }
        offset += idx * mat.strides()[i];
        ++i;
    }

    void *data_ptr = mat.storage()->data();
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        T *data      = static_cast<T *>(data_ptr);
        data[offset] = value.to<T>();
    });
}

}  // namespace cpu
}  // namespace origin
