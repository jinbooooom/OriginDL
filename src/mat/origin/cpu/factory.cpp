#include <memory>
#include <random>
#include <type_traits>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

namespace
{

/**
 * @brief 生成随机数的辅助函数
 * @tparam T 数据类型
 * @param data 数据指针
 * @param elements 元素数量
 * @param gen 随机数生成器
 */
template <typename T>
void fill_randn_impl(T *data, size_t elements, std::mt19937 &gen)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        // 浮点类型：直接使用对应类型的正态分布
        std::normal_distribution<T> dist(T(0), T(1));
        for (size_t i = 0; i < elements; ++i)
        {
            data[i] = dist(gen);
        }
    }
    else
    {
        // 整数类型：使用 float 分布然后转换
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < elements; ++i)
        {
            data[i] = static_cast<T>(dist(gen));
        }
    }
}

}  // namespace

std::unique_ptr<OriginMat> randn(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    std::random_device rd;
    std::mt19937 gen(rd());

    void *data = result->storage()->data();

    device_common::TypeDispatcher::dispatch_void(
        options.dtype(), [&]<typename T>() { fill_randn_impl<T>(static_cast<T *>(data), shape.elements(), gen); });

    return result;
}

std::unique_ptr<OriginMat> zeros(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        T *data = result->data_ptr<T>();
        for (size_t i = 0; i < shape.elements(); ++i)
        {
            data[i] = static_cast<T>(0);
        }
    });

    return result;
}

std::unique_ptr<OriginMat> ones(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        T *data = result->data_ptr<T>();
        for (size_t i = 0; i < shape.elements(); ++i)
        {
            data[i] = static_cast<T>(1);
        }
    });

    return result;
}

std::unique_ptr<OriginMat> full(const Shape &shape, const Scalar &scalar, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    // 使用类型分发器替代重复的switch语句
    device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename T>() {
        T *data = result->data_ptr<T>();
        T v     = scalar.to<T>();
        for (size_t i = 0; i < shape.elements(); ++i)
        {
            data[i] = v;
        }
    });

    return result;
}

std::unique_ptr<OriginMat> from_memory(const void *data,
                                       DataType user_dtype,
                                       const Shape &shape,
                                       const TensorOptions &options)
{
    // 创建存储
    size_t size  = shape.elements() * element_size(options.dtype());
    auto storage = Storage::create(size, options.device().type(), options.device().index());

    // 检查是否需要类型转换
    if (user_dtype == options.dtype())
    {
        // 不需要转换，直接内存复制
        memcpy(storage->data(), data, size);
    }
    else
    {
        // 需要类型转换，使用TypeDispatcher进行转换
        device_common::TypeDispatcher::dispatch_void(user_dtype, [&]<typename T>() {
            const T *user_data = static_cast<const T *>(data);
            // 使用TypeDispatcher进行目标类型转换
            device_common::TypeDispatcher::dispatch_void(options.dtype(), [&]<typename U>() {
                U *dst_data = static_cast<U *>(storage->data());
                for (size_t i = 0; i < shape.elements(); ++i)
                {
                    dst_data[i] = static_cast<U>(user_data[i]);
                }
            });
        });
    }

    return std::make_unique<OriginMat>(storage, shape, options.dtype());
}

}  // namespace cpu
}  // namespace origin
