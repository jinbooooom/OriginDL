#include <gtest/gtest.h>
#include <stdexcept>

/**
 * 简单的数组求和函数
 * 对一块int数组内存上的所有数据求和
 * @param data 指向int数组的指针
 * @param size 数组大小
 * @return 数组所有元素的和
 * @throws std::invalid_argument 当data为空指针时抛出异常
 */
int sum_array(const int* data, int size) {
    // 参数测试：检查空指针
    if (data == nullptr) {
        throw std::invalid_argument("Array pointer cannot be null");
    }
    
    // 参数测试：检查数组大小
    if (size < 0) {
        throw std::invalid_argument("Array size cannot be negative");
    }
    
    // 功能实现：计算数组元素的和
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
    
    return sum;
}

/**
 * GoogleTest 简单示例 - 数组求和测试
 * 
 * 这个文件演示了GoogleTest的基本用法：
 * 1. 参数测试 - 测试空指针异常
 * 2. 功能测试 - 测试求和结果是否正确
 */

// ==================== 参数测试用例 ====================

/**
 * 测试空指针异常
 * 当传入空指针时，函数应该抛出异常
 */
TEST(ArraySumTest, null_pointer_exception) {
    // 测试空指针异常
    EXPECT_THROW(sum_array(nullptr, 5), std::invalid_argument);
    
    // 测试空指针异常的具体错误信息
    try {
        sum_array(nullptr, 5);
        FAIL() << "Expected exception to be thrown, but none was thrown";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "Array pointer cannot be null");
    }
}

/**
 * 测试负数大小异常
 * 当传入负数大小时，函数应该抛出异常
 */
TEST(ArraySumTest, negative_size_exception) {
    int data[] = {1, 2, 3};
    
    // 测试负数大小异常
    EXPECT_THROW(sum_array(data, -1), std::invalid_argument);
    
    // 测试负数大小异常的具体错误信息
    try {
        sum_array(data, -1);
        FAIL() << "Expected exception to be thrown, but none was thrown";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "Array size cannot be negative");
    }
}

// ==================== 功能测试用例 ====================

/**
 * 测试基本求和功能
 * 测试数组 {0, 1, 2, 3} 的求和结果是否为 6
 */
TEST(ArraySumTest, basic_sum_functionality) {
    int data[] = {0, 1, 2, 3};
    int size = sizeof(data) / sizeof(data[0]);
    
    // 测试求和结果
    EXPECT_EQ(sum_array(data, size), 6);
}

/**
 * 测试空数组求和
 * 空数组的求和结果应该为 0
 */
TEST(ArraySumTest, empty_array_sum) {
    int data[] = {};
    int size = 0;
    
    // 空数组求和应该为 0
    EXPECT_EQ(sum_array(data, size), 0);
}

/**
 * 测试单个元素数组
 * 只有一个元素的数组求和
 */
TEST(ArraySumTest, single_element_array) {
    int data[] = {42};
    int size = 1;
    
    // 单个元素数组求和
    EXPECT_EQ(sum_array(data, size), 42);
}

// ==================== 边界测试用例 ====================

/**
 * 测试大数组求和
 * 测试较大数组的求和功能
 */
TEST(ArraySumTest, large_array_sum) {
    // 创建包含100个元素的数组，每个元素值为其索引
    int data[100];
    for (int i = 0; i < 100; ++i) {
        data[i] = i;
    }
    
    // 0+1+2+...+99 = 99*100/2 = 4950
    EXPECT_EQ(sum_array(data, 100), 4950);
}

// ==================== 主函数 ====================

int main(int argc, char **argv) {
    // 初始化GoogleTest
    ::testing::InitGoogleTest(&argc, argv);
    
    // 运行所有测试
    return RUN_ALL_TESTS();
}