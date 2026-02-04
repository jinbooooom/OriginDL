#ifndef __ORIGIN_UTILS_SIZE_ARRAY_REF_H__
#define __ORIGIN_UTILS_SIZE_ARRAY_REF_H__

#include <cstddef>
#include <vector>
#include <initializer_list>

namespace origin
{

/**
 * @brief 轻量级 size_t 数组引用类（仅用于函数参数传递）
 * @details 统一支持 C 数组、std::vector<size_t> 和 std::initializer_list<size_t>
 * 
 * 设计原则：
 * - 轻量级：只保存指针和大小，不拥有数据
 * - 只用于函数参数传递，不存储为成员变量
 * - 函数内部应立即调用 to_vector() 拷贝数据
 * - initializer_list 的生命周期会持续到函数调用结束，只要立即使用就是安全的
 * 
 * 使用示例：
 *   std::vector<size_t> vec = {1, 2, 3};
 *   split(x, vec, 0);  // 从 vector
 *   size_t arr[] = {1, 2, 3};
 *   split(x, arr, 0);  // 从 C 数组
 *   split(x, {1, 2, 3}, 0);  // 从 initializer_list（需要显式指定 size_t 类型）
 */
class SizeArrayRef {
private:
    const size_t* data_;
    size_t size_;
    
public:
    // 默认构造函数（空数组）
    constexpr SizeArrayRef() noexcept 
        : data_(nullptr), size_(0) {}
    
    // 从 C 数组构造
    template<size_t N>
    constexpr SizeArrayRef(const size_t (&arr)[N]) noexcept
        : data_(arr), size_(N) {}
    
    // 从指针和大小构造
    constexpr SizeArrayRef(const size_t* data, size_t size) noexcept
        : data_(data), size_(size) {}
    
    // 从 std::vector<size_t> 构造（零拷贝）
    SizeArrayRef(const std::vector<size_t>& vec) noexcept
        : data_(vec.data()), size_(vec.size()) {}
    
    // 从 std::initializer_list<size_t> 构造
    // 注意：initializer_list 的生命周期会持续到函数调用结束
    // 只要在函数调用时立即使用（如调用 to_vector()），就是安全的
    SizeArrayRef(std::initializer_list<size_t> list) noexcept
        : data_(list.begin()), size_(list.size()) {}
    
    // 明确拒绝其他类型的 vector
    template<typename T>
    SizeArrayRef(const std::vector<T>&) = delete;
    
    // 明确拒绝浮点数类型的 initializer_list
    SizeArrayRef(std::initializer_list<float>) = delete;
    SizeArrayRef(std::initializer_list<double>) = delete;
    
    // 访问接口
    constexpr const size_t* data() const noexcept { return data_; }
    constexpr size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    
    // 元素访问
    constexpr const size_t& operator[](size_t i) const noexcept {
        return data_[i];
    }
    
    const size_t& at(size_t i) const {
        if (i >= size_) {
            throw std::out_of_range("SizeArrayRef index out of range");
        }
        return data_[i];
    }
    
    // 迭代器接口
    constexpr const size_t* begin() const noexcept { return data_; }
    constexpr const size_t* end() const noexcept { return data_ + size_; }
    constexpr const size_t* cbegin() const noexcept { return data_; }
    constexpr const size_t* cend() const noexcept { return data_ + size_; }
    
    // 转换为 vector
    std::vector<size_t> to_vector() const {
        return std::vector<size_t>(data_, data_ + size_);
    }
    
    // 比较操作
    bool operator==(const SizeArrayRef& other) const noexcept {
        if (size_ != other.size_) return false;
        for (size_t i = 0; i < size_; ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }
    
    bool operator!=(const SizeArrayRef& other) const noexcept {
        return !(*this == other);
    }
};

}  // namespace origin

#endif  // __ORIGIN_UTILS_SIZE_ARRAY_REF_H__
