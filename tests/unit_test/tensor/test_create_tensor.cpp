#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
using namespace origin;

class TensorCreateTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}

    // 辅助函数：比较两个浮点数是否相等（考虑浮点精度）
    bool isEqual(double a, double b, double tolerance = 1e-6) { return std::abs(a - b) < tolerance; }
};

// 从向量构造张量测试
TEST_F(TensorCreateTest, ConstructorFromVector)
{
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 4U);

    auto result_data = tensor.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], data[i], 1e-6);
    }
}

// 从初始化列表构造张量测试
TEST_F(TensorCreateTest, ConstructorFromInitializerList)
{
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});

    Shape expected_shape{2, 2};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.elements(), 4U);

    auto result_data             = tensor.to_vector<float>();
    std::vector<data_t> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], 1e-6);
    }
}

// 从标量构造张量测试
TEST_F(TensorCreateTest, ConstructorFromScalar)
{
    data_t value = 5.0;
    Shape shape{3, 3};
    Tensor tensor(value, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 9U);

    auto result_data = tensor.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], value, 1e-6);
    }
}

// 拷贝构造函数测试
TEST_F(TensorCreateTest, CopyConstructor)
{
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};
    Tensor original(data, shape);
    Tensor copy(original);

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.elements(), original.elements());

    auto original_data = original.to_vector<float>();
    auto copy_data     = copy.to_vector<float>();
    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(copy_data[i], original_data[i], 1e-6);
    }
}

// 移动构造函数测试
TEST_F(TensorCreateTest, MoveConstructor)
{
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};
    Tensor original_tensor(data, shape);

    // 保存原始数据，用于后续验证
    auto original_data = original_tensor.to_vector<float>();

    // 执行移动构造
    Tensor moved_tensor(std::move(original_tensor));

    // 验证移动后的对象数据正确
    EXPECT_EQ(moved_tensor.shape(), shape);
    EXPECT_EQ(moved_tensor.elements(), 4U);

    auto moved_data = moved_tensor.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(moved_data[i], data[i], 1e-6);
    }

    // 验证移动构造的核心特性：原始对象应该处于无效状态
    // 由于Tensor使用shared_ptr，移动后original_tensor.impl_变为nullptr
    // 这是正确的移动语义行为：所有权被转移，原对象不再拥有资源

    // 验证移动构造确实发生了：尝试访问移动后的对象应该导致段错误
    // 这证明了移动构造函数的正确实现
    // 注意：这个测试期望程序崩溃，这是正确的行为
    // 在实际使用中，不应该访问移动后的对象

    // 下面的代码来验证移动构造
    // EXPECT_DEATH 会启动一个子进程来执行测试，由于访问 nullptr，子进程产生段错误，父进程捕获子进程的崩溃信号
    EXPECT_DEATH(
        {
            printf("access moved object...\n");
            fflush(stdout);
            original_tensor.shape();
            printf("this message should not print\n");
        },
        ".*");
    EXPECT_DEATH(original_tensor.elements(), ".*");   // 期望段错误
    EXPECT_DEATH(original_tensor.to_vector<float>(), ".*");  // 期望段错误

    // 验证多次移动构造的正确性
    Tensor another_tensor = std::move(moved_tensor);
    EXPECT_EQ(another_tensor.shape(), shape);
    EXPECT_EQ(another_tensor.elements(), 4U);

    auto another_data = another_tensor.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(another_data[i], data[i], 1e-6);
    }

    Tensor final_tensor = std::move(another_tensor);
    EXPECT_EQ(final_tensor.shape(), shape);
    EXPECT_EQ(final_tensor.elements(), 4U);

    auto final_data = final_tensor.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(final_data[i], data[i], 1e-6);
    }
}

// 赋值运算符测试
TEST_F(TensorCreateTest, AssignmentOperators)
{
    std::vector<data_t> data1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_t> data2 = {5.0, 6.0, 7.0, 8.0};
    Shape shape{2, 2};

    Tensor tensor1(data1, shape);
    Tensor tensor2(data2, shape);

    // 拷贝赋值
    tensor2 = tensor1;
    EXPECT_EQ(tensor2.shape(), tensor1.shape());
    EXPECT_EQ(tensor2.elements(), tensor1.elements());

    auto data1_vec = tensor1.to_vector<float>();
    auto data2_vec = tensor2.to_vector<float>();
    for (size_t i = 0; i < data1_vec.size(); ++i)
    {
        EXPECT_NEAR(data2_vec[i], data1_vec[i], 1e-6);
    }

    // 移动赋值
    Tensor tensor3(data2, shape);
    tensor3 = std::move(tensor1);
    EXPECT_EQ(tensor3.shape(), shape);
    EXPECT_EQ(tensor3.elements(), 4U);
}

// 工厂方法测试
TEST_F(TensorCreateTest, FactoryMethods)
{
    Shape shape{3, 3};

    // zeros
    Tensor zeros_tensor = Tensor::zeros(shape);
    EXPECT_EQ(zeros_tensor.shape(), shape);
    auto zeros_data = zeros_tensor.to_vector<float>();
    for (size_t i = 0; i < zeros_data.size(); ++i)
    {
        EXPECT_NEAR(zeros_data[i], 0.0, 1e-6);
    }

    // ones
    Tensor ones_tensor = Tensor::ones(shape);
    EXPECT_EQ(ones_tensor.shape(), shape);
    auto ones_data = ones_tensor.to_vector<float>();
    for (size_t i = 0; i < ones_data.size(); ++i)
    {
        EXPECT_NEAR(ones_data[i], 1.0, 1e-6);
    }

    // constant
    data_t value           = 2.5;
    Tensor constant_tensor = Tensor(value, shape);
    EXPECT_EQ(constant_tensor.shape(), shape);
    auto constant_data = constant_tensor.to_vector<float>();
    for (size_t i = 0; i < constant_data.size(); ++i)
    {
        EXPECT_NEAR(constant_data[i], value, 1e-6);
    }
}

// 随机张量工厂测试
TEST_F(TensorCreateTest, RandnFactory)
{
    Shape shape{2, 2};
    Tensor rand_tensor = Tensor::randn(shape);

    EXPECT_EQ(rand_tensor.shape(), shape);
    EXPECT_EQ(rand_tensor.elements(), 4U);

    // 随机数应该在合理范围内
    auto rand_data = rand_tensor.to_vector<float>();
    for (size_t i = 0; i < rand_data.size(); ++i)
    {
        EXPECT_TRUE(std::abs(rand_data[i]) < 10.0);  // 大部分随机数应该在[-10, 10]范围内
    }
}

// 形状验证测试
TEST_F(TensorCreateTest, ShapeValidation)
{
    // 测试零维度
    Shape zero_shape{0};
    std::vector<data_t> data = {1.0};
    EXPECT_THROW(Tensor tensor(data, zero_shape), std::invalid_argument);

    // 测试数据大小不匹配
    Shape valid_shape{2, 2};
    std::vector<data_t> small_data = {1.0};  // 只有1个元素，但形状需要4个元素
    EXPECT_THROW(Tensor tensor(small_data, valid_shape), std::invalid_argument);

    // 测试有效形状
    std::vector<data_t> valid_data = {1.0, 2.0, 3.0, 4.0};
    EXPECT_NO_THROW(Tensor tensor(valid_data, valid_shape));
}

// 空张量测试
TEST_F(TensorCreateTest, EmptyTensor)
{
    std::vector<data_t> empty_data;
    Shape empty_shape{0};

    EXPECT_THROW(Tensor tensor(empty_data, empty_shape), std::invalid_argument);
}

// 标量张量测试
TEST_F(TensorCreateTest, ScalarTensor)
{
    std::vector<data_t> data = {42.0};
    Shape shape{1};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.elements(), 1U);
    EXPECT_NEAR(tensor.item<float>(), 42.0, 1e-6);
}

// 大张量测试
TEST_F(TensorCreateTest, LargeTensor)
{
    size_t size = 1000;
    std::vector<data_t> data(size, 1.0);
    Shape shape{size};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), size);

    auto result_data = tensor.to_vector<float>();
    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_NEAR(result_data[i], 1.0, 1e-6);
    }
}

// 一维张量测试
TEST_F(TensorCreateTest, OneDimensionalTensor)
{
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    Shape shape{5};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.ndim(), 1U);
    EXPECT_EQ(tensor.elements(), 5U);

    auto result_data = tensor.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], data[i], 1e-6);
    }
}

// 三维张量测试
TEST_F(TensorCreateTest, ThreeDimensionalTensor)
{
    std::vector<data_t> data(24, 1.0);  // 2*3*4 = 24
    Shape shape{2, 3, 4};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.ndim(), 3U);
    EXPECT_EQ(tensor.elements(), 24U);

    auto result_data = tensor.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], data[i], 1e-6);
    }
}

// 数据完整性测试
TEST_F(TensorCreateTest, DataIntegrity)
{
    std::vector<data_t> original_data = {1.1, 2.2, 3.3, 4.4, 5.5};
    Shape shape{5, 1};
    Tensor tensor(original_data, shape);

    auto retrieved_data = tensor.to_vector<float>();
    EXPECT_EQ(original_data.size(), retrieved_data.size());
    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(original_data[i], retrieved_data[i], 1e-6);
    }
}

// 内存管理测试
TEST_F(TensorCreateTest, MemoryManagement)
{
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};

    {
        Tensor tensor1(data, shape);
        Tensor tensor2 = tensor1;             // 拷贝构造
        Tensor tensor3 = std::move(tensor1);  // 移动构造

        EXPECT_EQ(tensor2.shape(), shape);
        EXPECT_EQ(tensor3.shape(), shape);

        auto data2 = tensor2.to_vector<float>();
        auto data3 = tensor3.to_vector<float>();
        for (size_t i = 0; i < data.size(); ++i)
        {
            EXPECT_NEAR(data2[i], data[i], 1e-6);
            EXPECT_NEAR(data3[i], data[i], 1e-6);
        }
    }  // 离开作用域后，张量应该被正确销毁
}