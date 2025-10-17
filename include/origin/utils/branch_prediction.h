/**
 * @file branch_prediction.h
 * @brief 分支预测优化宏定义
 * 
 * 这个头文件提供了用于分支预测优化的宏，帮助编译器生成更优化的代码。
 * 在性能关键的代码路径中使用这些宏可以显著提升执行效率。
 */

#ifndef __ORIGIN_DL_BRANCH_PREDICTION_H__
#define __ORIGIN_DL_BRANCH_PREDICTION_H__

/**
 * @brief 标记条件为很可能为真
 * 
 * 使用 __builtin_expect 告诉编译器这个条件很可能为真，
 * 编译器会优化代码布局，将很可能执行的分支放在更优的位置。
 * 
 * @param x 要评估的条件表达式
 * @return 条件表达式的值
 * 
 * @example
 * if (likely(x > 0)) {
 *     // 这个分支很可能被执行，编译器会优化代码布局
 *     process_positive(x);
 * }
 */
#define likely(x) __builtin_expect(!!(x), 1)

/**
 * @brief 标记条件为很可能为假
 * 
 * 使用 __builtin_expect 告诉编译器这个条件很可能为假，
 * 编译器会优化代码布局，将很可能执行的分支放在更优的位置。
 * 
 * @param x 要评估的条件表达式
 * @return 条件表达式的值
 * 
 * @example
 * if (unlikely(error_condition)) {
 *     // 这个分支很少被执行，编译器会优化代码布局
 *     handle_error();
 * }
 */
#define unlikely(x) __builtin_expect(!!(x), 0)


#endif // __ORIGIN_DL_BRANCH_PREDICTION_H__
