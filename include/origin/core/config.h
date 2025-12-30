#ifndef __ORIGIN_DL_CONFIG_H__
#define __ORIGIN_DL_CONFIG_H__

namespace origin
{

/**
 * @brief 全局配置命名空间
 */
namespace Config
{
/**
 * @brief 是否启用反向传播
 * @details 当为false时，backward()不会构建计算图
 */
extern bool enable_backprop;
}  // namespace Config

/**
 * @brief no_grad() 返回的RAII guard对象
 *
 * 用于在作用域内禁用梯度计算
 * 用法：
 * {
 *     auto guard = no_grad();
 *     // 在这个作用域内，梯度计算被禁用
 * }  // guard析构时自动恢复
 */
class NoGradGuard
{
private:
    bool old_value_;  // 保存旧的值

public:
    /**
     * @brief 构造函数：禁用梯度计算
     */
    NoGradGuard();

    /**
     * @brief 析构函数：恢复梯度计算
     */
    ~NoGradGuard();

    // 禁止拷贝，允许移动
    NoGradGuard(const NoGradGuard &)            = delete;
    NoGradGuard &operator=(const NoGradGuard &) = delete;
    NoGradGuard(NoGradGuard &&)                 = default;
    NoGradGuard &operator=(NoGradGuard &&)      = default;
};

/**
 * @brief 禁用梯度计算的上下文管理器
 *
 * 返回一个RAII guard对象，在作用域内禁用梯度计算
 * 与 PyTorch 的 torch.no_grad() 行为一致
 *
 * @return NoGradGuard对象，析构时自动恢复
 *
 * @example
 * {
 *     auto guard = no_grad();
 *     auto y = model(x);  // 不会构建计算图
 *     auto loss = criterion(y, t);
 * }  // guard析构，恢复梯度计算
 */
NoGradGuard no_grad();

}  // namespace origin

#endif  // __ORIGIN_DL_CONFIG_H__
