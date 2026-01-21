#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;

/**
 * @brief 张量索引操作测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 */
class TensorIndexTest : public origin::test::OperatorTestBase
{};

// ==================== index() 读取测试 ====================

TEST_P(TensorIndexTest, IndexReadBasic)
{
    // 创建3维张量 {3, 4, 5}
    std::vector<float> data(3 * 4 * 5);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }
    Tensor tensor = Tensor(data, Shape{3, 4, 5}, dtype(DataType::kFloat32).device(deviceType()));

    // 测试读取元素
    float value = tensor.index<float>({1, 2, 3});
    // 计算期望值：1*4*5 + 2*5 + 3 = 20 + 10 + 3 = 33
    EXPECT_FLOAT_EQ(value, 33.0f);
}

TEST_P(TensorIndexTest, IndexReadFirstElement)
{
    // 创建2维张量
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    Tensor tensor = Tensor(data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 读取第一个元素
    float value = tensor.index<float>({0, 0});
    EXPECT_FLOAT_EQ(value, 1.0f);
}

TEST_P(TensorIndexTest, IndexReadLastElement)
{
    // 创建2维张量
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    Tensor tensor = Tensor(data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 读取最后一个元素
    float value = tensor.index<float>({1, 1});
    EXPECT_FLOAT_EQ(value, 4.0f);
}

TEST_P(TensorIndexTest, IndexReadDifferentTypes)
{
    // 测试不同数据类型
    std::vector<int32_t> int_data{10, 20, 30, 40};
    Tensor int_tensor = Tensor(int_data, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));

    int32_t int_value = int_tensor.index<int32_t>({1, 0});
    EXPECT_EQ(int_value, 30);

    std::vector<double> double_data{1.5, 2.5, 3.5, 4.5};
    Tensor double_tensor = Tensor(double_data, Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));

    double double_value = double_tensor.index<double>({0, 1});
    EXPECT_DOUBLE_EQ(double_value, 2.5);
}

// ==================== index_put() 写入测试 ====================

TEST_P(TensorIndexTest, IndexPutBasic)
{
    // 创建3维张量
    std::vector<float> data(3 * 4 * 5, 0.0f);
    Tensor tensor = Tensor(data, Shape{3, 4, 5}, dtype(DataType::kFloat32).device(deviceType()));

    // 写入元素
    tensor.index_put({1, 2, 3}, 42.0f);

    // 读取验证
    float value = tensor.index<float>({1, 2, 3});
    EXPECT_FLOAT_EQ(value, 42.0f);
}

TEST_P(TensorIndexTest, IndexPutMultiple)
{
    // 创建2维张量
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    Tensor tensor = Tensor(data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 写入多个元素
    tensor.index_put({0, 0}, 10.0f);
    tensor.index_put({0, 1}, 20.0f);
    tensor.index_put({1, 0}, 30.0f);
    tensor.index_put({1, 1}, 40.0f);

    // 验证
    EXPECT_FLOAT_EQ(tensor.index<float>({0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(tensor.index<float>({0, 1}), 20.0f);
    EXPECT_FLOAT_EQ(tensor.index<float>({1, 0}), 30.0f);
    EXPECT_FLOAT_EQ(tensor.index<float>({1, 1}), 40.0f);
}

TEST_P(TensorIndexTest, IndexPutDifferentTypes)
{
    // 测试不同数据类型
    std::vector<int32_t> int_data{1, 2, 3, 4};
    Tensor int_tensor = Tensor(int_data, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));

    int_tensor.index_put({1, 1}, 100);
    EXPECT_EQ(int_tensor.index<int32_t>({1, 1}), 100);

    std::vector<double> double_data{1.0, 2.0, 3.0, 4.0};
    Tensor double_tensor = Tensor(double_data, Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));

    double_tensor.index_put({0, 0}, 99.5);
    EXPECT_DOUBLE_EQ(double_tensor.index<double>({0, 0}), 99.5);
}

// ==================== 视图（View）测试 ====================

TEST_P(TensorIndexTest, IndexWithView)
{
    // 创建原始张量
    std::vector<float> data(3 * 4 * 5);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }
    Tensor original = Tensor(data, Shape{3, 4, 5}, dtype(DataType::kFloat32).device(deviceType()));

    // 创建视图
    Tensor view = original.reshape(Shape{12, 5});

    // 在视图中读取元素
    // view[6, 2] 对应 original 的线性索引 6*5 + 2 = 32
    float view_value = view.index<float>({6, 2});
    EXPECT_FLOAT_EQ(view_value, 32.0f);

    // 在视图中写入元素
    view.index_put({6, 2}, 999.0f);

    // 验证原始张量也被修改（视图共享存储）
    // 实际上 view[6,2] 对应 original 的索引需要重新计算
    // 6*5 + 2 = 32，在原始shape {3,4,5}中：32 / (4*5) = 1, (32 % (4*5)) / 5 = 2, 32 % 5 = 2
    // 所以是 original[1, 2, 2]
    float check_value = original.index<float>({1, 2, 2});
    EXPECT_FLOAT_EQ(check_value, 999.0f);
}

// ==================== 转置（非连续内存）测试 ====================
// TODO: 当前转置是数据转置（重新排列数据），不是视图转置（共享存储），以后可能会改正为视图转置

TEST_P(TensorIndexTest, IndexWithTranspose)
{
    // 创建2维张量 {3, 4}
    // 数据按行存储：[[0,1,2,3], [4,5,6,7], [8,9,10,11]]
    std::vector<float> data(3 * 4);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }
    Tensor original = Tensor(data, Shape{3, 4}, dtype(DataType::kFloat32).device(deviceType()));

    // 转置后形状变为 {4, 3}，内存不连续
    Tensor transposed = original.transpose();

    // 验证转置后的形状
    EXPECT_EQ(transposed.shape().size(), 2);
    EXPECT_EQ(transposed.shape()[0], 4);
    EXPECT_EQ(transposed.shape()[1], 3);

    // 在转置后的张量上读取元素
    // original[1, 2] = 1*4 + 2 = 6
    // transposed[2, 1] 应该等于 original[1, 2] = 6
    float value = transposed.index<float>({2, 1});
    EXPECT_FLOAT_EQ(value, 6.0f);

    // 验证其他位置
    // original[0, 0] = 0, transposed[0, 0] = 0
    EXPECT_FLOAT_EQ(transposed.index<float>({0, 0}), 0.0f);
    // original[2, 3] = 2*4 + 3 = 11, transposed[3, 2] = 11
    EXPECT_FLOAT_EQ(transposed.index<float>({3, 2}), 11.0f);
}

TEST_P(TensorIndexTest, IndexPutWithTranspose)
{
    // 创建2维张量 {2, 3}
    // 数据：[[1,2,3], [4,5,6]]，按行存储为 [1,2,3,4,5,6]
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor original = Tensor(data, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 转置后形状变为 {3, 2}，数据重新排列为 [[1,4], [2,5], [3,6]]，即 [1,4,2,5,3,6]
    Tensor transposed = original.transpose();

    // 验证转置后的数据
    // transposed[0, 0] = 1.0f (对应 original[0, 0])
    EXPECT_FLOAT_EQ(transposed.index<float>({0, 0}), 1.0f);
    // transposed[0, 1] = 4.0f (对应 original[1, 0])
    EXPECT_FLOAT_EQ(transposed.index<float>({0, 1}), 4.0f);
    // transposed[1, 0] = 2.0f (对应 original[0, 1])
    EXPECT_FLOAT_EQ(transposed.index<float>({1, 0}), 2.0f);

    // 在转置后的张量上写入元素
    // transposed[1, 0] 对应 original[0, 1]，写入后应该影响转置后的数据
    transposed.index_put({1, 0}, 99.0f);

    // 验证转置后的张量
    EXPECT_FLOAT_EQ(transposed.index<float>({1, 0}), 99.0f);

    // TODO: 当前转置是数据转置（重新排列数据），不是视图转置（共享存储），所以原始张量不会被修改
    // 如果改为视图转置，原始张量也会被修改
    // 验证原始张量未被修改
    EXPECT_FLOAT_EQ(original.index<float>({0, 1}), 2.0f);

    // 再写入另一个位置
    transposed.index_put({2, 1}, 88.0f);
    EXPECT_FLOAT_EQ(transposed.index<float>({2, 1}), 88.0f);
    // 原始张量未被修改
    EXPECT_FLOAT_EQ(original.index<float>({1, 2}), 6.0f);
}

TEST_P(TensorIndexTest, IndexPutWithTranspose3D)
{
    // 创建3维张量 {2, 3, 4}
    std::vector<float> data(2 * 3 * 4);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }
    Tensor original = Tensor(data, Shape{2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));

    // 转置最后两个维度，形状变为 {2, 4, 3}
    Tensor transposed = original.transpose();

    // 在转置后的张量上读取
    // original[1, 2, 3] = 1*3*4 + 2*4 + 3 = 12 + 8 + 3 = 23
    // 转置后，original[1, 2, 3] 对应 transposed[1, 3, 2]
    float read_value = transposed.index<float>({1, 3, 2});
    EXPECT_FLOAT_EQ(read_value, 23.0f);

    // 写入转置后的张量
    transposed.index_put({1, 3, 2}, 777.0f);

    // 验证写入成功
    EXPECT_FLOAT_EQ(transposed.index<float>({1, 3, 2}), 777.0f);
    // TODO: 当前转置是数据转置（重新排列数据），不是视图转置（共享存储），所以原始张量不会被修改
    // 如果改为视图转置，原始张量也会被修改
    // 验证原始张量未被修改
    EXPECT_FLOAT_EQ(original.index<float>({1, 2, 3}), 23.0f);
}

TEST_P(TensorIndexTest, IndexWithDoubleTranspose)
{
    // 测试双重转置（应该恢复原状）
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor original = Tensor(data, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    Tensor transposed1 = original.transpose();
    Tensor transposed2 = transposed1.transpose();

    // 双重转置后应该恢复原状
    EXPECT_EQ(transposed2.shape()[0], original.shape()[0]);
    EXPECT_EQ(transposed2.shape()[1], original.shape()[1]);

    // 验证值也恢复
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            float orig_val = original.index<float>({i, j});
            float trans_val = transposed2.index<float>({i, j});
            EXPECT_FLOAT_EQ(orig_val, trans_val);
        }
    }

    // 在双重转置后的张量上写入
    transposed2.index_put({1, 2}, 999.0f);
    EXPECT_FLOAT_EQ(transposed2.index<float>({1, 2}), 999.0f);
    // TODO: 当前转置是数据转置（重新排列数据），不是视图转置（共享存储），所以原始张量不会被修改
    // 如果改为视图转置，原始张量也会被修改
    EXPECT_FLOAT_EQ(original.index<float>({1, 2}), 6.0f);
}

// ==================== 边界检查测试 ====================

TEST_P(TensorIndexTest, IndexOutOfRange)
{
    Tensor tensor = Tensor::ones(Shape{3, 4, 5}, dtype(DataType::kFloat32).device(deviceType()));

    // 测试索引超出范围（实际抛出 std::invalid_argument）
    EXPECT_THROW({ tensor.index<float>({3, 0, 0}); }, std::exception);  // 第一维超出
    EXPECT_THROW({ tensor.index<float>({0, 4, 0}); }, std::exception);  // 第二维超出
    EXPECT_THROW({ tensor.index<float>({0, 0, 5}); }, std::exception);  // 第三维超出
}

TEST_P(TensorIndexTest, IndexWrongDimension)
{
    Tensor tensor = Tensor::ones(Shape{3, 4, 5}, dtype(DataType::kFloat32).device(deviceType()));

    // 测试索引数量不匹配（实际抛出 std::invalid_argument）
    EXPECT_THROW({ tensor.index<float>({1, 2}); }, std::exception);      // 索引太少
    EXPECT_THROW({ tensor.index<float>({1, 2, 3, 4}); }, std::exception); // 索引太多
}

TEST_P(TensorIndexTest, IndexPutOutOfRange)
{
    Tensor tensor = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 测试写入索引超出范围（实际抛出 std::invalid_argument）
    EXPECT_THROW({ tensor.index_put({2, 0}, 1.0f); }, std::exception);
    EXPECT_THROW({ tensor.index_put({0, 2}, 1.0f); }, std::exception);
}

// ==================== 参数化测试实例化 ====================

INSTANTIATE_TEST_SUITE_P(TensorIndexTests, TensorIndexTest, ::testing::Values(DeviceType::kCPU),
                         [](const ::testing::TestParamInfo<DeviceType> &info) {
                             return info.param == DeviceType::kCPU ? "CPU" : "CUDA";
                         });

#ifdef WITH_CUDA
INSTANTIATE_TEST_SUITE_P(TensorIndexTestsCUDA, TensorIndexTest, ::testing::Values(DeviceType::kCUDA),
                         [](const ::testing::TestParamInfo<DeviceType> &info) {
                             return info.param == DeviceType::kCPU ? "CPU" : "CUDA";
                         });
#endif
