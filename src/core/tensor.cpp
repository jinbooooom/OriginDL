#include "origin/core/tensor.h"
#include <cstring>
#include <stdexcept>
#include "origin/core/config.h"
#include "origin/core/tensor_options.h"
#include "origin/mat/backend.h"
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{

// === 构造函数和析构函数实现 ===

// 内部构造函数实现
Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

// 拷贝构造函数并不会深拷贝，只是共享底层数据
Tensor::Tensor(const Tensor &other) : impl_(other.impl_) {}

// 移动构造函数会转移所有权，原对象变为无效状态
Tensor::Tensor(Tensor &&other) noexcept : impl_(std::move(other.impl_)) {}

// 拷贝赋值运算符并不会深拷贝，只是共享底层数据，原对象保持不变
Tensor &Tensor::operator=(const Tensor &other)
{
    if (this != &other)
    {
        impl_ = other.impl_;
    }
    return *this;
}

// 移动赋值运算符会转移所有权，原对象变为无效状态，当前对象获得所有权
Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this != &other)
    {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

// 从Mat创建Tensor的构造函数实现
Tensor::Tensor(std::unique_ptr<Mat> mat) : impl_(std::make_shared<TensorImpl>(std::move(mat))) {}

// === 工厂方法实现 ===

Tensor Tensor::zeros(const Shape &shape, const TensorOptions &options)
{
    return Tensor(0, shape, options);
}

Tensor Tensor::ones(const Shape &shape, const TensorOptions &options)
{
    return Tensor(1, shape, options);
}

Tensor Tensor::randn(const Shape &shape, const TensorOptions &options)
{
    auto impl = TensorImpl::randn(shape, options);
    return Tensor(std::make_shared<TensorImpl>(std::move(impl)));
}

Tensor Tensor::full(const Shape &shape, const Scalar &value, const TensorOptions &options)
{
    return Tensor(value, shape, options);
}

Tensor Tensor::from_blob(void *data, const Shape &shape, const TensorOptions &options)
{
    Tensor result;
    result.from_memory(data, options.dtype(), shape, options);

    return result;
}

// === 形状和维度实现 ===

Shape Tensor::shape() const
{
    return impl_->shape();
}

size_t Tensor::ndim() const
{
    return impl_->ndim();
}

size_t Tensor::elements() const
{
    return impl_->elements();
}

// === 张量属性方法实现 ===

size_t Tensor::element_size() const
{
    return origin::element_size(dtype());  // 返回单个元素占用的字节数
}

size_t Tensor::numel() const
{
    return elements();  // numel()和elements()功能相同
}

size_t Tensor::nbytes() const
{
    return element_size() * numel();
}

// === 数据访问：类型安全实现 ===

template <typename T>
T Tensor::item() const
{
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    return impl_->item<T>();
}

template <typename T>
T *Tensor::data_ptr()
{
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    return impl_->data_ptr<T>();
}

// === 索引访问实现 ===

template <typename T>
T Tensor::index(std::initializer_list<size_t> indices) const
{
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    Scalar result = impl_->index(indices);
    return result.to<T>();
}

void Tensor::index_put(std::initializer_list<size_t> indices, const Scalar &value)
{
    impl_->index_put(indices, value);
}

// === 类型查询和转换实现 ===

DataType Tensor::dtype() const
{
    return impl_->data_->dtype();
}

Device Tensor::device() const
{
    return impl_->data_->device();
}

Tensor Tensor::to(DataType target_type) const
{
    auto converted_mat = impl_->data_->to(target_type);
    return Tensor(std::make_unique<TensorImpl>(std::move(converted_mat)));
}

Tensor Tensor::to(Device device) const
{
    auto converted_mat = impl_->data_->to_device(device);
    return Tensor(std::make_unique<TensorImpl>(std::move(converted_mat)));
}

Tensor Tensor::to(const TensorOptions &options) const
{
    auto converted_impl = impl_->to(options);
    return Tensor(std::make_shared<TensorImpl>(std::move(converted_impl)));
}

// === 梯度相关实现 ===

Tensor Tensor::grad() const
{
    if (!impl_->grad_)
    {
        return Tensor::zeros(shape(),
                             origin::dtype(DataType::kFloat32).device(device()));  // TODO，创建与input同类型的gtad
    }
    // 返回共享的梯度（与PyTorch行为一致）
    auto grad_impl = std::make_shared<TensorImpl>(impl_->grad_);  // 使用 shared_ptr 构造函数，共享 grad_
    return Tensor(grad_impl);
}

void Tensor::set_creator(const std::shared_ptr<Operator> &func)
{
    impl_->set_creator(func);
}

void Tensor::backward()
{
    impl_->backward();
}

void Tensor::clear_grad()
{
    impl_->clear_grad();
}

bool Tensor::requires_grad() const
{
    // TODO: jinbo 当前的origindl不支持requires_grad=false，所以默认是true，未来支持后，需要修改
    return true;  // Config::enable_backprop && impl_ && impl_->creator_ != nullptr;
}

Tensor Tensor::detach() const
{
    // 创建一个新的TensorImpl，只复制data_，不复制creator_和grad_
    // 这样新tensor就不会参与计算图，可以安全释放
    // 注意：原始tensor保持不变（因为detach是const方法）
    // 对于clone().detach()，clone()返回的中间tensor会在超出作用域时自动释放
    auto new_impl = std::make_shared<TensorImpl>(impl_->data_);
    return Tensor(new_impl);
}

Tensor Tensor::clone() const
{
    // 1. 深拷贝data_（创建独立的数据副本）
    // 2. 不复制grad_（初始化为nullptr，需要重新计算梯度）
    // 3. 复制creator_和generation_（保留计算图连接，仍可参与梯度计算）
    auto cloned_data = impl_->data_ ? impl_->data_->clone() : nullptr;
    auto new_impl    = std::make_shared<TensorImpl>(std::move(cloned_data));
    // 复制计算图信息
    new_impl->creator_    = impl_->creator_;
    new_impl->generation_ = impl_->generation_;
    return Tensor(new_impl);
}

void Tensor::accumulate_grad(const Tensor &grad_to_add)
{
    // 如果梯度为空，直接赋值；否则累加（类似backward()中的实现）
    if (!impl_->grad_)
    {
        impl_->grad_ = grad_to_add.impl_->data_;
    }
    else
    {
        // 梯度不为空，原地累加（不创建新的 Mat，减少内存分配）
        impl_->grad_->add_inplace(*grad_to_add.impl_->data_);
    }
}

// === 张量操作实现 ===

Tensor Tensor::reshape(const Shape &shape) const
{
    // 通过TensorImpl的reshape方法，避免直接操作Mat
    auto new_impl = impl_->reshape(shape);
    return Tensor(std::make_shared<TensorImpl>(std::move(new_impl)));
}

Tensor Tensor::transpose() const
{
    // 通过TensorImpl的transpose方法，避免直接操作Mat
    auto new_impl = impl_->transpose();
    return Tensor(std::make_shared<TensorImpl>(std::move(new_impl)));
}

Tensor Tensor::contiguous() const
{
    // 通过 Mat 接口的 contiguous() 创建连续张量
    auto new_mat = impl_->data_->contiguous();
    return Tensor(std::move(new_mat));
}

// === 调试实现 ===
void Tensor::print(const std::string &desc) const
{
    impl_->print(desc);
}

namespace to_vector_detail
{
template <typename T>
struct ToVectorConvert
{
    Tensor &t;
    std::vector<T> &result;
    size_t n;
    template <typename SrcT>
    void operator()()
    {
        const SrcT *p = t.data_ptr<SrcT>();
        for (size_t i = 0; i < n; ++i)
            result[i] = static_cast<T>(p[i]);
    }
};
}  // namespace to_vector_detail

// 这个函数就调式的时候用，在业务中尽量少用，因为会创建新的vector，耗时慢。
// 如果只是为了访问数据，尽量使用item()或者data_ptr()方法。
template <typename T>
std::vector<T> Tensor::to_vector() const
{
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);

    // 1. 获取在内存中连续存储的张量
    Tensor t = contiguous();

    // 2. 确保在 CPU 上（具体同步和拷贝由后端的 to()/to_device() 负责）
    if (t.device().type() != DeviceType::kCPU)
    {
        t = t.to(Device(DeviceType::kCPU));
    }

    size_t n = t.elements();
    std::vector<T> result(n);

    if (t.dtype() == DataTypeTraits<T>::type)
    {
        // 类型一致：直接 memcpy，无需逐元素转换
        const T *src = t.data_ptr<T>();
        std::memcpy(result.data(), src, t.nbytes());
        return result;
    }

    // 类型不一致：按张量实际 dtype 分发，逐元素 static_cast 到 T
    device_common::TypeDispatcher::dispatch_void(t.dtype(), to_vector_detail::ToVectorConvert<T>{t, result, n});

    return result;
}

int Tensor::backend_type() const
{
    return impl_->backend_type();
}

// === 私有方法实现 ===

void Tensor::from_memory(const void *data, DataType user_dtype, const Shape &shape, const TensorOptions &options)
{
    // 验证形状是否有效
    // 0维张量（标量张量）是合法的，但其他维度不能为0
    if (!shape.is_scalar())
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (unlikely(shape[i] == 0))
            {
                THROW_INVALID_ARG("Tensor shape cannot have zero dimensions. Dimension {} is zero in shape {}", i,
                                  shape.to_string());
            }
        }
    }

    // 直接调用TensorImpl工厂方法
    impl_ = std::make_unique<TensorImpl>(TensorImpl::from_memory(data, user_dtype, shape, options));
}

void Tensor::from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options)
{
    // 验证形状是否有效
    // 0维张量（标量张量）是合法的，但其他维度不能为0
    if (!shape.is_scalar())
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (unlikely(shape[i] == 0))
            {
                THROW_INVALID_ARG("Tensor shape cannot have zero dimensions. Dimension {} is zero in shape {}", i,
                                  shape.to_string());
            }
        }
    }

    // 直接调用TensorImpl工厂方法并设置impl_
    impl_ = std::make_unique<TensorImpl>(TensorImpl::from_scalar(scalar, shape, options));
}

// === 模板实例化 ===

// 数据访问方法
template float Tensor::item<float>() const;
template double Tensor::item<double>() const;
template int32_t Tensor::item<int32_t>() const;
template int8_t Tensor::item<int8_t>() const;
template unsigned long Tensor::item<unsigned long>() const;
// 新增：支持 uint8_t / int64_t（通常为 unsigned char / long）标量访问
template unsigned char Tensor::item<unsigned char>() const;
template long Tensor::item<long>() const;

template float *Tensor::data_ptr<float>();
template double *Tensor::data_ptr<double>();
template int32_t *Tensor::data_ptr<int32_t>();
template int8_t *Tensor::data_ptr<int8_t>();
template unsigned long *Tensor::data_ptr<unsigned long>();
// 新增：支持 uint8_t / int64_t（通常为 unsigned char / long）数据指针访问
template unsigned char *Tensor::data_ptr<unsigned char>();
template long *Tensor::data_ptr<long>();

// 索引访问方法
template float Tensor::index<float>(std::initializer_list<size_t>) const;
template double Tensor::index<double>(std::initializer_list<size_t>) const;
template int32_t Tensor::index<int32_t>(std::initializer_list<size_t>) const;
template int8_t Tensor::index<int8_t>(std::initializer_list<size_t>) const;
template unsigned long Tensor::index<unsigned long>(std::initializer_list<size_t>) const;

template std::vector<float> Tensor::to_vector<float>() const;
template std::vector<double> Tensor::to_vector<double>() const;
template std::vector<int32_t> Tensor::to_vector<int32_t>() const;
template std::vector<int8_t> Tensor::to_vector<int8_t>() const;

}  // namespace origin