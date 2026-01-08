#ifndef __ORIGIN_DL_CONV_UTILS_H__
#define __ORIGIN_DL_CONV_UTILS_H__

#include <utility>

namespace origin
{

/**
 * @brief 将单个值或pair转换为pair
 * @param value 单个int值或pair<int, int>
 * @return pair<int, int>
 */
inline std::pair<int, int> pair(int value)
{
    return std::make_pair(value, value);
}

inline std::pair<int, int> pair(std::pair<int, int> value)
{
    return value;
}

/**
 * @brief 计算卷积输出尺寸
 * @param input_size 输入尺寸
 * @param kernel_size 卷积核尺寸
 * @param stride 步长
 * @param pad 填充
 * @return 输出尺寸
 */
inline int get_conv_outsize(int input_size, int kernel_size, int stride, int pad)
{
    return (input_size + pad * 2 - kernel_size) / stride + 1;
}

}  // namespace origin

#endif  // __ORIGIN_DL_CONV_UTILS_H__
