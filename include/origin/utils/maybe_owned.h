#ifndef __ORIGIN_DL_MAYBE_OWNED_H__
#define __ORIGIN_DL_MAYBE_OWNED_H__

#include <memory>
#include <type_traits>
#include "origin/core/tensor.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{

/**
 * @brief MaybeOwned 智能指针，用于优化类型提升场景
 * @details 参考 PyTorch 的 c10::MaybeOwned 设计
 *          - 如果类型匹配，只借用引用（borrowed），不增加引用计数
 *          - 如果类型不匹配，拥有新创建的 tensor（owned）
 *
 * 使用场景：
 *   auto [x0, x1] = TypePromotion::promote_tensors_maybe_owned(a, b);
 *   如果 a 和 b 类型相同，x0 和 x1 只是借用，零开销
 *   如果类型不同，只有需要转换的 tensor 才会创建新对象
 */
template <typename T>
class MaybeOwned
{
public:
    /**
     * @brief 借用构造函数：从引用创建，不拥有所有权
     * @param ref 要借用的对象引用
     * @return MaybeOwned 对象，借用模式
     * @details 零开销，只存储指针，不增加引用计数
     */
    static MaybeOwned borrowed(const T &ref) { return MaybeOwned(&ref, false); }

    /**
     * @brief 拥有构造函数：从右值创建，拥有所有权
     * @param value 要拥有的对象（右值）
     * @return MaybeOwned 对象，拥有模式
     * @details 创建新对象并拥有所有权，析构时自动释放
     */
    static MaybeOwned owned(T &&value) { return MaybeOwned(std::make_unique<T>(std::move(value))); }

    /**
     * @brief 拥有构造函数：从值创建（拷贝），拥有所有权
     * @param value 要拥有的对象（左值）
     * @return MaybeOwned 对象，拥有模式
     * @details 拷贝对象并拥有所有权，析构时自动释放
     */
    static MaybeOwned owned(const T &value) { return MaybeOwned(std::make_unique<T>(value)); }

    /**
     * @brief 默认构造函数：创建空对象
     */
    MaybeOwned() : ptr_(nullptr), is_owned_(false) {}

    /**
     * @brief 移动构造函数
     * @param other 要移动的对象
     */
    MaybeOwned(MaybeOwned &&other) noexcept
        : ptr_(other.ptr_), owned_ptr_(std::move(other.owned_ptr_)), is_owned_(other.is_owned_)
    {
        other.ptr_      = nullptr;
        other.is_owned_ = false;
    }

    /**
     * @brief 拷贝构造函数
     * @param other 要拷贝的对象
     * @details 如果是借用的，只复制指针；如果是拥有的，创建新的拥有对象
     */
    MaybeOwned(const MaybeOwned &other) : is_owned_(other.is_owned_)
    {
        if (other.is_owned_)
        {
            owned_ptr_ = std::make_unique<T>(*other.owned_ptr_);
            ptr_       = owned_ptr_.get();
        }
        else
        {
            ptr_ = other.ptr_;
        }
    }

    /**
     * @brief 移动赋值运算符
     * @param other 要移动的对象
     * @return 当前对象的引用
     */
    MaybeOwned &operator=(MaybeOwned &&other) noexcept
    {
        if (this != &other)
        {
            ptr_            = other.ptr_;
            owned_ptr_      = std::move(other.owned_ptr_);
            is_owned_       = other.is_owned_;
            other.ptr_      = nullptr;
            other.is_owned_ = false;
        }
        return *this;
    }

    /**
     * @brief 拷贝赋值运算符
     * @param other 要拷贝的对象
     * @return 当前对象的引用
     */
    MaybeOwned &operator=(const MaybeOwned &other)
    {
        if (this != &other)
        {
            is_owned_ = other.is_owned_;
            if (other.is_owned_)
            {
                owned_ptr_ = std::make_unique<T>(*other.owned_ptr_);
                ptr_       = owned_ptr_.get();
            }
            else
            {
                ptr_ = other.ptr_;
            }
        }
        return *this;
    }

    /**
     * @brief 析构函数
     * @details 如果是借用的，不需要释放；如果是拥有的，unique_ptr 会自动释放
     */
    ~MaybeOwned() = default;

    /**
     * @brief 解引用运算符：返回引用
     * @return 指向对象的引用
     * @throws 如果指针为空，抛出运行时异常
     */
    T &operator*() const
    {
        if (unlikely(!ptr_))
        {
            THROW_RUNTIME_ERROR("MaybeOwned: dereferencing null pointer");
        }
        return *ptr_;
    }

    /**
     * @brief 成员访问运算符
     * @return 指向对象的指针
     * @throws 如果指针为空，抛出运行时异常
     */
    T *operator->() const
    {
        if (unlikely(!ptr_))
        {
            THROW_RUNTIME_ERROR("MaybeOwned: accessing null pointer");
        }
        return ptr_;
    }

    /**
     * @brief 获取原始指针
     * @return 指向对象的指针，可能为空
     */
    T *get() const { return ptr_; }

    /**
     * @brief 检查是否为空
     * @return true 如果指针为空，false 否则
     */
    [[nodiscard]] bool is_null() const { return ptr_ == nullptr; }

    /**
     * @brief 检查是否拥有所有权
     * @return true 如果拥有所有权，false 否则
     */
    [[nodiscard]] bool is_owned() const { return is_owned_; }

    /**
     * @brief 检查是否是借用的
     * @return true 如果是借用模式，false 否则
     */
    [[nodiscard]] bool is_borrowed() const { return !is_owned_ && ptr_ != nullptr; }

    /**
     * @brief 隐式转换为 T&（用于支持参数是Tensor& 的函数，比如 mat(Tensor& tensor) 等函数）
     * @return 指向对象的引用
     * @details 保持隐式转换以支持 mat(x0) 等便捷用法
     * @note 注意生命周期：如果 MaybeOwned 是借用的，确保原对象在使用期间有效
     */
    operator T &() const { return *ptr_; }

    // /**
    //  * @brief 隐式转换为 T*（用于需要指针的场景）
    //  * @return 指向对象的指针
    //  * @note 已注释：使用场景不明确，且存在悬空指针风险
    //  *       如需指针，请使用 get() 方法显式获取
    //  */
    // operator T *() const { return ptr_; }

private:
    /**
     * @brief 借用构造函数（私有）
     * @param ptr 要借用的对象指针
     * @param dummy 占位参数，用于区分构造函数
     */
    MaybeOwned(const T *ptr, bool) : ptr_(const_cast<T *>(ptr)), is_owned_(false) {}

    /**
     * @brief 拥有构造函数（私有）
     * @param owned 要拥有的对象（unique_ptr）
     */
    explicit MaybeOwned(std::unique_ptr<T> owned) : owned_ptr_(std::move(owned)), is_owned_(true)
    {
        ptr_ = owned_ptr_.get();
    }

    T *ptr_;                        ///< 指向对象的指针（可能是借用的或拥有的）
    std::unique_ptr<T> owned_ptr_;  ///< 拥有的对象（仅在 is_owned_ 为 true 时有效）
    bool is_owned_;                 ///< 是否拥有所有权
};

}  // namespace origin

#endif  // __ORIGIN_DL_MAYBE_OWNED_H__
