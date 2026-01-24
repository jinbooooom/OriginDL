#include "origin/core/tensor_impl.h"
#include <functional>
#include <list>
#include <set>
#include <stdexcept>
#include "origin/core/config.h"
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/core/tensor_options.h"
#include "origin/mat/backend.h"
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{

// 两个核心工厂方法实现
TensorImpl TensorImpl::from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options)
{
    // 通过后端Mat接口工厂方法（与后端解耦）
    auto mat = Mat_t::from_scalar(scalar, shape, options);
    return TensorImpl(std::move(mat));
}

TensorImpl TensorImpl::from_memory(const void *data,
                                   DataType user_dtype,
                                   const Shape &shape,
                                   const TensorOptions &options)
{
    // 通过后端Mat接口工厂方法（与后端解耦）
    auto mat = Mat_t::from_memory(data, user_dtype, shape, options);
    return TensorImpl(std::move(mat));
}

// 静态工厂方法实现
TensorImpl TensorImpl::randn(const Shape &shape)
{
    // 通过后端Mat接口创建随机数矩阵
    auto mat = Mat_t::randn(shape);
    return TensorImpl(std::move(mat));
}

TensorImpl TensorImpl::randn(const Shape &shape, const TensorOptions &options)
{
    // 通过后端Mat接口创建随机数矩阵
    auto mat = Mat_t::randn(shape, options);
    return TensorImpl(std::move(mat));
}

// 赋值运算符实现 - clone data_ 和 grad_（保证值语义，赋值后独立）
TensorImpl &TensorImpl::operator=(const TensorImpl &other)
{
    if (this != &other)
    {
        data_       = other.data_ ? std::shared_ptr<Mat>(other.data_->clone()) : nullptr;
        grad_       = other.grad_ ? std::shared_ptr<Mat>(other.grad_->clone()) : nullptr;
        creator_    = other.creator_;
        generation_ = other.generation_;
    }
    return *this;
}

TensorImpl &TensorImpl::operator=(TensorImpl &&other) noexcept
{
    if (this != &other)
    {
        data_       = std::move(other.data_);
        grad_       = std::move(other.grad_);
        creator_    = std::move(other.creator_);
        generation_ = other.generation_;
    }
    return *this;
}

void TensorImpl::set_creator(const std::shared_ptr<Operator> &func)
{
    creator_    = func;
    generation_ = creator_->generation_ + 1;
}

void TensorImpl::backward()
{
    // 检查是否启用反向传播
    if (!Config::enable_backprop)
    {
        return;  // 禁用梯度计算，直接返回
    }

    // 如果梯度为空，初始化为全1（输出张量的梯度）
    // 梯度类型应与数据类型一致，设备应与数据设备一致
    if (!grad_)
    {
        auto data_device      = data_->device();
        TensorOptions options = TensorOptions(data_->dtype()).device(data_device);
        grad_                 = std::shared_ptr<Mat>(Mat_t::from_scalar(1, data_->shape(), options));
    }

    auto funcs    = std::list<std::shared_ptr<Operator>>();
    auto func_set = std::set<std::shared_ptr<Operator>>();

    auto add_func = [&funcs, &func_set](const std::shared_ptr<Operator> &f) {
        if (f && func_set.find(f) == func_set.end())
        {
            funcs.push_back(f);
            func_set.insert(f);
            funcs.sort([](const std::shared_ptr<Operator> &lhs, const std::shared_ptr<Operator> &rhs) {
                return lhs->generation_ < rhs->generation_;
            });
        }
    };

    add_func(this->creator_);

    // 记录已处理的Operator和tensor，用于后续清理
    std::vector<std::shared_ptr<Operator>> processed_ops;
    std::set<std::shared_ptr<TensorImpl>> processed_tensors;

    while (!funcs.empty())
    {
        auto f = funcs.back();
        funcs.pop_back();

        auto gys = std::vector<Tensor>();
        // 检查 outputs_ 是否为空
        if (unlikely(f->outputs_.empty()))
        {
            // 如果outputs_为空，跳过这个Operator
            continue;
        }

        // 记录这个Operator已被处理
        processed_ops.push_back(f);

        // 收集所有输出tensor的impl_，用于后续清理
        // 将 weak_ptr 转换为 shared_ptr（如果有效）
        std::vector<std::shared_ptr<TensorImpl>>
            valid_outputs;  // 临时保存有效的 shared_ptr，确保在 backward() 期间有效
        for (const auto &weak_output : f->outputs_)
        {
            // 将 weak_ptr 转换为 shared_ptr
            auto output_impl = weak_output.lock();
            if (!output_impl)
            {
                // weak_ptr 失效，说明用户代码中的 tensor 已经超出作用域
                // 这是正常的，跳过这个输出
                continue;
            }

            // 临时保存有效的 shared_ptr，确保在 backward() 期间有效
            valid_outputs.push_back(output_impl);

            // 记录tensor的impl_，用于后续清理
            processed_tensors.insert(output_impl);

            // 获取输出张量的梯度
            Tensor output_tensor(output_impl);
            gys.push_back(output_tensor.grad());
        }

        // 如果所有 weak_ptr 都失效，跳过这个 Operator
        if (unlikely(gys.empty()))
        {
            continue;
        }
        auto gxs = f->backward(gys);

        if (unlikely(gxs.size() != f->inputs_.size()))
        {
            THROW_RUNTIME_ERROR("backward error!, gxs size {} inputs size {}", gxs.size(), f->inputs_.size());
        }

        for (size_t i = 0; i < gxs.size(); i++)
        {
            auto x  = f->inputs_[i];
            auto gx = gxs[i];

            // 记录输入tensor的impl_，用于后续清理
            if (x.impl_)
            {
                processed_tensors.insert(x.impl_);
            }

            // 梯度累积逻辑：如果梯度为空，直接赋值；否则累加
            if (!x.impl_->grad_)
            {
                // 梯度为空，直接共享（底层返回 unique_ptr，转换为 shared_ptr）
                x.impl_->grad_ = gx.impl_->data_;
            }
            else
            {
                // 梯度不为空，原地累加（不创建新的 Mat，减少内存分配）
                x.impl_->grad_->add_inplace(*gx.impl_->data_);
            }

            if (x.impl_->creator_)
            {
                add_func(x.impl_->creator_);
            }
        }
    }

    // backward()完成后，自动清理计算图以释放内存
    //
    // 关键设计（使用 weak_ptr 后）：
    // 1. outputs_ 使用 weak_ptr，不再持有强引用，不会造成循环引用
    // 2. 先收集输入tensor信息（在清理inputs_之前）
    // 3. 清理所有涉及的Operator的inputs_，减少内存占用
    // 4. 清理所有相关tensor的grad_和creator_，彻底断开循环引用并释放内存

    // 第一步：收集输入tensor信息（在清理inputs_之前）
    std::set<std::shared_ptr<TensorImpl>> input_tensors;
    for (const auto &f : processed_ops)
    {
        if (f)
        {
            for (const auto &input : f->inputs_)
            {
                if (input.impl_)
                {
                    input_tensors.insert(input.impl_);
                }
            }
        }
    }

    // 第二步：清理Operator的inputs_（减少内存占用）
    // 注意：outputs_ 使用 weak_ptr，不再持有强引用，不会造成循环引用，不需要清理
    for (const auto &f : processed_ops)
    {
        if (f)
        {
            // 清理inputs_，减少内存占用
            // inputs_是值语义的Tensor，清理后不会影响其他引用
            f->inputs_.clear();
        }
    }

    // 第三步：清理所有相关tensor的grad_和creator_，彻底断开循环引用并释放内存
    // 注意：只清理在当前backward()调用中涉及的tensor，不会影响其他tensor
    //
    // 关键设计：
    // 1. 清理中间tensor的grad_和creator_，释放梯度内存并断开循环引用
    // 2. 输出tensor（this）的grad_需要保留（用户可能需要），但清理creator_
    // 3. 输入tensor的grad_需要保留（用户可能需要），但清理creator_
    // 4. 中间tensor的grad_在backward()完成后就不再需要，可以安全清理
    bool is_output_tensor = false;
    for (const auto &tensor_impl : processed_tensors)
    {
        if (tensor_impl)
        {
            // 判断是否是输出tensor（当前backward()的起点）
            is_output_tensor = (tensor_impl.get() == this);

            // 判断是否是输入tensor（需要保留grad_）
            bool is_input_tensor = (input_tensors.find(tensor_impl) != input_tensors.end());

            // 清理creator_，断开循环引用
            tensor_impl->creator_    = nullptr;
            tensor_impl->generation_ = 0;

            // 清理中间tensor的grad_，释放梯度内存
            // 输入tensor和输出tensor的grad_需要保留，供用户使用
            if (!is_input_tensor && !is_output_tensor)
            {
                // 中间tensor：清理grad_，释放梯度内存
                tensor_impl->grad_ = nullptr;
            }
        }
    }

    // 清理当前tensor的creator_，断开与计算图的连接
    // 这样可以确保整个计算图都可以被释放
    this->creator_    = nullptr;
    this->generation_ = 0;
}

void TensorImpl::clear_grad()
{
    grad_ = nullptr;
}

void TensorImpl::detach()
{
    // 断开与计算图的连接，递归清理整个计算图
    // 这样可以彻底断开循环引用，释放GPU内存
    if (!creator_)
    {
        return;  // 已经detach过了
    }

    // 收集所有相关的Operator和tensor，递归清理整个计算图
    std::set<std::shared_ptr<Operator>> processed_ops;
    std::set<std::shared_ptr<TensorImpl>> processed_tensors;

    std::function<void(const std::shared_ptr<Operator> &)> collect_ops = [&](const std::shared_ptr<Operator> &op) {
        if (!op || processed_ops.find(op) != processed_ops.end())
        {
            return;
        }
        processed_ops.insert(op);

        // 收集所有输出tensor（将 weak_ptr 转换为 shared_ptr）
        for (const auto &weak_output : op->outputs_)
        {
            auto output_impl = weak_output.lock();
            if (output_impl)
            {
                processed_tensors.insert(output_impl);
            }
        }

        // 递归收集输入tensor的creator
        for (const auto &input : op->inputs_)
        {
            if (input.impl_)
            {
                processed_tensors.insert(input.impl_);
                if (input.impl_->creator_)
                {
                    collect_ops(input.impl_->creator_);
                }
            }
        }
    };

    // 从当前tensor的creator开始收集
    collect_ops(creator_);

    // 清理所有收集到的Operator的outputs_和inputs_，断开循环引用
    for (const auto &op : processed_ops)
    {
        op->outputs_.clear();
        op->inputs_.clear();
    }

    // 清理所有相关tensor的creator_，彻底断开循环引用
    for (const auto &tensor_impl : processed_tensors)
    {
        if (tensor_impl)
        {
            tensor_impl->creator_    = nullptr;
            tensor_impl->generation_ = 0;
            // 也清理grad_，释放梯度内存
            tensor_impl->grad_ = nullptr;
        }
    }

    // 断开当前tensor的creator_
    creator_    = nullptr;
    generation_ = 0;
    grad_       = nullptr;  // 也清理当前tensor的梯度
}

TensorImpl TensorImpl::reshape(const Shape &shape) const
{
    auto new_mat = data_->reshape(shape);
    return TensorImpl(std::move(new_mat));  // 构造函数会自动将 unique_ptr 转换为 shared_ptr
}

TensorImpl TensorImpl::transpose() const
{
    auto new_mat = data_->transpose();
    return TensorImpl(std::move(new_mat));  // 构造函数会自动将 unique_ptr 转换为 shared_ptr
}

// 访问器方法实现
Shape TensorImpl::shape() const
{
    return data_->shape();
}

size_t TensorImpl::ndim() const
{
    return data_->shape().size();
}

size_t TensorImpl::elements() const
{
    return data_->elements();
}

template <typename T>
T TensorImpl::item() const
{
    if (unlikely(elements() != 1))
    {
        THROW_RUNTIME_ERROR("item() can only be called on scalar tensors, but tensor has {} elements", elements());
    }
    return data_->to_vector<T>()[0];
}

template <typename T>
std::vector<T> TensorImpl::to_vector() const
{
    return data_->to_vector<T>();
}

// === 泛型数据访问方法实现 ===

template <typename T>
T *TensorImpl::data_ptr()
{
    // TensorImpl::data_ptr<T>() 调用 data_->data_ptr()（虚函数）
    // 对于 OriginMat，会调用 OriginMat::data_ptr() 的虚函数版本，返回 void*
    // 然后转换为 T*
    // 这样 TensorImpl 使用的是虚函数版本，而不是模板版本。
    // 模板版本 template <typename T> T *data_ptr() 保留给内部实现代码（如 cpu/ 和 cuda/
    // 目录下的文件）直接使用，提供类型安全。
    return static_cast<T *>(data_->data_ptr());
}

// === 索引访问实现 ===

Scalar TensorImpl::index(std::initializer_list<size_t> indices) const
{
    return data_->index(indices);
}

void TensorImpl::index_put(std::initializer_list<size_t> indices, const Scalar &value)
{
    data_->index_put(indices, value);
}

// === 泛型标量操作实现 ===
// 标量运算符已移除，统一通过算子层处理

int TensorImpl::backend_type() const
{
    return data_->backend_type();
}

// 调试方法实现
void TensorImpl::print(const std::string &desc) const
{
    if (data_)
    {
        data_->print(desc);
    }
}

// 类型转换实现
TensorImpl TensorImpl::to(const TensorOptions &options) const
{
    auto converted_mat = data_->to(options.dtype());
    // 如果目标设备与当前设备不同，需要进行设备转换
    if (converted_mat->device() != options.device())
    {
        converted_mat = converted_mat->to_device(options.device());
    }
    return TensorImpl(std::move(converted_mat));  // 构造函数会自动将 unique_ptr 转换为 shared_ptr
}

// 移除所有私有辅助方法，直接实现核心逻辑

// === 泛型方法实例化 ===
// 数据访问方法
template float TensorImpl::item<float>() const;
template double TensorImpl::item<double>() const;
template int32_t TensorImpl::item<int32_t>() const;
template int8_t TensorImpl::item<int8_t>() const;

template float *TensorImpl::data_ptr<float>();
template double *TensorImpl::data_ptr<double>();
template int32_t *TensorImpl::data_ptr<int32_t>();
template int8_t *TensorImpl::data_ptr<int8_t>();

template std::vector<float> TensorImpl::to_vector<float>() const;
template std::vector<double> TensorImpl::to_vector<double>() const;
template std::vector<int32_t> TensorImpl::to_vector<int32_t>() const;
template std::vector<int8_t> TensorImpl::to_vector<int8_t>() const;

// 泛型标量操作已移除，统一通过算子层处理

// 额外的模板实例化（只添加新的类型）
template unsigned long TensorImpl::item<unsigned long>() const;
template unsigned long *TensorImpl::data_ptr<unsigned long>();
template std::vector<unsigned long> TensorImpl::to_vector<unsigned long>() const;

}  // namespace origin