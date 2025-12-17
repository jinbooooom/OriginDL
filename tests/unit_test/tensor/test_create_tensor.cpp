#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;

class TensorCreateTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// 从向量构造张量测试
TEST_F(TensorCreateTest, ConstructorFromVector)
{
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Shape shape{2, 2};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 4U);

    auto expected = Tensor(data, shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor, expected, origin::test::TestTolerance::kDefault);
}

// 从初始化列表构造张量测试
TEST_F(TensorCreateTest, ConstructorFromInitializerList)
{
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});

    Shape expected_shape{2, 2};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.elements(), 4U);

    auto expected = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, expected_shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor, expected, origin::test::TestTolerance::kDefault);
}

// 从标量构造张量测试
TEST_F(TensorCreateTest, ConstructorFromScalar)
{
    float value = 5.0f;
    Shape shape{3, 3};
    Tensor tensor(value, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 9U);

    auto expected = Tensor(std::vector<float>(9, value), shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor, expected, origin::test::TestTolerance::kDefault);
}

// 拷贝构造函数测试
TEST_F(TensorCreateTest, CopyConstructor)
{
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Shape shape{2, 2};
    Tensor original(data, shape);
    Tensor copy(original);

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.elements(), original.elements());

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(copy, original, origin::test::TestTolerance::kDefault);
}

// 移动构造函数测试
TEST_F(TensorCreateTest, MoveConstructor)
{
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Shape shape{2, 2};
    Tensor original_tensor(data, shape);

    // 保存原始数据，用于后续验证
    auto original_data = original_tensor.to_vector<float>();

    // 执行移动构造
    Tensor moved_tensor(std::move(original_tensor));

    // 验证移动后的对象数据正确
    EXPECT_EQ(moved_tensor.shape(), shape);
    EXPECT_EQ(moved_tensor.elements(), 4U);

    auto expected = Tensor(data, shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(moved_tensor, expected, origin::test::TestTolerance::kDefault);

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
    EXPECT_DEATH(original_tensor.elements(), ".*");          // 期望段错误
    EXPECT_DEATH(original_tensor.to_vector<float>(), ".*");  // 期望段错误

    // 验证多次移动构造的正确性
    Tensor another_tensor = std::move(moved_tensor);
    EXPECT_EQ(another_tensor.shape(), shape);
    EXPECT_EQ(another_tensor.elements(), 4U);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(another_tensor, expected, origin::test::TestTolerance::kDefault);

    Tensor final_tensor = std::move(another_tensor);
    EXPECT_EQ(final_tensor.shape(), shape);
    EXPECT_EQ(final_tensor.elements(), 4U);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(final_tensor, expected, origin::test::TestTolerance::kDefault);
}

// 赋值运算符测试
TEST_F(TensorCreateTest, AssignmentOperators)
{
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    Shape shape{2, 2};

    Tensor tensor1(data1, shape);
    Tensor tensor2(data2, shape);

    // 拷贝赋值
    tensor2 = tensor1;
    EXPECT_EQ(tensor2.shape(), tensor1.shape());
    EXPECT_EQ(tensor2.elements(), tensor1.elements());

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor2, tensor1, origin::test::TestTolerance::kDefault);

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
    Tensor zeros_tensor = Tensor::zeros(shape, dtype(DataType::kFloat32));
    EXPECT_EQ(zeros_tensor.shape(), shape);
    auto expected_zeros = Tensor::zeros(shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(zeros_tensor, expected_zeros, origin::test::TestTolerance::kDefault);

    // ones
    Tensor ones_tensor = Tensor::ones(shape, dtype(DataType::kFloat32));
    EXPECT_EQ(ones_tensor.shape(), shape);
    auto expected_ones = Tensor::ones(shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(ones_tensor, expected_ones, origin::test::TestTolerance::kDefault);

    // constant
    float value           = 2.5f;
    Tensor constant_tensor = Tensor(value, shape);
    EXPECT_EQ(constant_tensor.shape(), shape);
    auto expected_constant = Tensor(std::vector<float>(9, value), shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(constant_tensor, expected_constant, origin::test::TestTolerance::kDefault);
}

// 随机张量工厂测试
TEST_F(TensorCreateTest, RandnFactory)
{
    Shape shape{2, 2};
    Tensor rand_tensor = Tensor::randn(shape, dtype(DataType::kFloat32));

    EXPECT_EQ(rand_tensor.shape(), shape);
    EXPECT_EQ(rand_tensor.elements(), 4U);

    // 随机数应该在合理范围内
    auto rand_data = rand_tensor.to_vector<float>();
    for (size_t i = 0; i < rand_data.size(); ++i)
    {
        EXPECT_TRUE(std::abs(rand_data[i]) < 10.0f);  // 大部分随机数应该在[-10, 10]范围内
    }
}

// 形状验证测试
TEST_F(TensorCreateTest, ShapeValidation)
{
    // 测试零维度
    Shape zero_shape{0};
    std::vector<float> data = {1.0f};
    EXPECT_THROW(Tensor tensor(data, zero_shape), std::invalid_argument);

    // 测试数据大小不匹配
    Shape valid_shape{2, 2};
    std::vector<float> small_data = {1.0f};  // 只有1个元素，但形状需要4个元素
    EXPECT_THROW(Tensor tensor(small_data, valid_shape), std::invalid_argument);

    // 测试有效形状
    std::vector<float> valid_data = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_NO_THROW(Tensor tensor(valid_data, valid_shape));
}

// 空张量测试
TEST_F(TensorCreateTest, EmptyTensor)
{
    std::vector<float> empty_data;
    Shape empty_shape{0};

    EXPECT_THROW(Tensor tensor(empty_data, empty_shape), std::invalid_argument);
}

// 标量张量测试
TEST_F(TensorCreateTest, ScalarTensor)
{
    std::vector<float> data = {42.0f};
    Shape shape{1};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.elements(), 1U);
    EXPECT_NEAR(tensor.item<float>(), 42.0f, origin::test::TestTolerance::kDefault);
}

// 大张量测试
TEST_F(TensorCreateTest, LargeTensor)
{
    size_t size = 1000;
    std::vector<float> data(size, 1.0f);
    Shape shape{size};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), size);

    auto expected = Tensor(data, shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor, expected, origin::test::TestTolerance::kDefault);
}

// 一维张量测试
TEST_F(TensorCreateTest, OneDimensionalTensor)
{
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Shape shape{5};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.ndim(), 1U);
    EXPECT_EQ(tensor.elements(), 5U);

    auto expected = Tensor(data, shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor, expected, origin::test::TestTolerance::kDefault);
}

// 三维张量测试
TEST_F(TensorCreateTest, ThreeDimensionalTensor)
{
    std::vector<float> data(24, 1.0f);  // 2*3*4 = 24
    Shape shape{2, 3, 4};
    Tensor tensor(data, shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.ndim(), 3U);
    EXPECT_EQ(tensor.elements(), 24U);

    auto expected = Tensor(data, shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor, expected, origin::test::TestTolerance::kDefault);
}

// 数据完整性测试
TEST_F(TensorCreateTest, DataIntegrity)
{
    std::vector<float> original_data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    Shape shape{5, 1};
    Tensor tensor(original_data, shape);

    auto expected = Tensor(original_data, shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor, expected, origin::test::TestTolerance::kDefault);
}

// 内存管理测试
TEST_F(TensorCreateTest, MemoryManagement)
{
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Shape shape{2, 2};

    {
        Tensor tensor1(data, shape);
        Tensor tensor2 = tensor1;             // 拷贝构造
        Tensor tensor3 = std::move(tensor1);  // 移动构造

        EXPECT_EQ(tensor2.shape(), shape);
        EXPECT_EQ(tensor3.shape(), shape);

        auto expected = Tensor(data, shape, dtype(DataType::kFloat32));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor2, expected, origin::test::TestTolerance::kDefault);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor3, expected, origin::test::TestTolerance::kDefault);
    }  // 离开作用域后，张量应该被正确销毁
}

// 测试张量创建的内存生命周期问题
TEST_F(TensorCreateTest, TensorMemoryLifecycle)
{
    // 模拟可能的内存生命周期问题
    Tensor tensor1, tensor2, tensor3;
    {
        // 在作用域内创建数据
        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> data3 = {9.0f, 10.0f, 11.0f, 12.0f};

        // 创建张量
        tensor1 = Tensor(data1, Shape{2, 2}, Float32);
        tensor2 = Tensor(data2, Shape{2, 2}, Float32);
        tensor3 = Tensor(data3, Shape{2, 2}, Float32);

        // 验证数据正确性
        auto expected1 = Tensor(data1, Shape{2, 2}, dtype(DataType::kFloat32));
        auto expected2 = Tensor(data2, Shape{2, 2}, dtype(DataType::kFloat32));
        auto expected3 = Tensor(data3, Shape{2, 2}, dtype(DataType::kFloat32));

        origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor1, expected1, origin::test::TestTolerance::kDefault);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor2, expected2, origin::test::TestTolerance::kDefault);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor3, expected3, origin::test::TestTolerance::kDefault);

    }  // 数据离开作用域，但张量应该仍然有效

    // 再次验证数据正确性
    std::vector<float> expected1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected2 = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> expected3 = {9.0f, 10.0f, 11.0f, 12.0f};

    auto exp1 = Tensor(expected1, Shape{2, 2}, dtype(DataType::kFloat32));
    auto exp2 = Tensor(expected2, Shape{2, 2}, dtype(DataType::kFloat32));
    auto exp3 = Tensor(expected3, Shape{2, 2}, dtype(DataType::kFloat32));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor1, exp1, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor2, exp2, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(tensor3, exp3, origin::test::TestTolerance::kDefault);
}

// 测试不同数据类型的张量创建
TEST_F(TensorCreateTest, DifferentDataTypeCreation)
{
    // 测试 float 类型
    std::vector<float> float_data = {0.0f, 1.0f, 2.0f, 3.0f};
    auto float_tensor             = Tensor(float_data, Shape{2, 2}, Float32);
    auto float_expected           = Tensor(float_data, Shape{2, 2}, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(float_tensor, float_expected, origin::test::TestTolerance::kDefault);

    // 测试 int 类型
    std::vector<int32_t> int_data = {0, 1, 2, 3};
    auto int_tensor               = Tensor(int_data, Shape{2, 2}, Int32);
    auto int_result               = int_tensor.to_vector<int32_t>();

    for (size_t i = 0; i < int_data.size(); ++i)
    {
        EXPECT_EQ(int_result[i], int_data[i]);
    }
}

// ==================== Tensor::full() 测试 ====================

TEST_F(TensorCreateTest, FullFactoryMethod)
{
    // 测试基本full方法
    Shape shape{2, 2};
    float value = 2.5f;
    auto t = Tensor::full(shape, value, dtype(DataType::kFloat32));
    
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.elements(), 4U);
    EXPECT_EQ(t.dtype(), DataType::kFloat32);
    
    auto expected = Tensor(std::vector<float>(4, value), shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(t, expected, origin::test::TestTolerance::kDefault);
}

TEST_F(TensorCreateTest, FullDifferentValues)
{
    // 测试不同填充值
    Shape shape{3, 1};
    
    // 正数
    auto t1 = Tensor::full(shape, 5.0f, dtype(DataType::kFloat32));
    auto expected1 = Tensor(std::vector<float>(3, 5.0f), shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(t1, expected1, origin::test::TestTolerance::kDefault);
    
    // 负数
    auto t2 = Tensor::full(shape, -1.0f, dtype(DataType::kFloat32));
    auto expected2 = Tensor(std::vector<float>(3, -1.0f), shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(t2, expected2, origin::test::TestTolerance::kDefault);
    
    // 零
    auto t3 = Tensor::full(shape, 0.0f, dtype(DataType::kFloat32));
    auto expected3 = Tensor::zeros(shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(t3, expected3, origin::test::TestTolerance::kDefault);
}

TEST_F(TensorCreateTest, FullDifferentTypes)
{
    // 测试不同数据类型的full
    Shape shape{2, 2};
    
    // float32
    auto t_float32 = Tensor::full(shape, 2.5f, dtype(DataType::kFloat32));
    EXPECT_EQ(t_float32.dtype(), DataType::kFloat32);
    
    // float64
    auto t_float64 = Tensor::full(shape, 2.5, dtype(DataType::kFloat64));
    EXPECT_EQ(t_float64.dtype(), DataType::kFloat64);
    
    // int32
    auto t_int32 = Tensor::full(shape, 42, dtype(DataType::kInt32));
    EXPECT_EQ(t_int32.dtype(), DataType::kInt32);
    auto int_data = t_int32.to_vector<int32_t>();
    for (size_t i = 0; i < int_data.size(); ++i)
    {
        EXPECT_EQ(int_data[i], 42);
    }
}

// ==================== Tensor::from_blob() 测试 ====================

TEST_F(TensorCreateTest, FromBlobBasic)
{
    // 测试基本from_blob方法
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Shape shape{2, 2};
    auto t = Tensor::from_blob(data, shape, dtype(DataType::kFloat32));
    
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.elements(), 4U);
    EXPECT_EQ(t.dtype(), DataType::kFloat32);
    
    auto expected = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(t, expected, origin::test::TestTolerance::kDefault);
}

TEST_F(TensorCreateTest, FromBlobDifferentShapes)
{
    // 测试不同形状的from_blob
    // 1维
    float data1[] = {1.0f, 2.0f, 3.0f};
    auto t1 = Tensor::from_blob(data1, Shape{3}, dtype(DataType::kFloat32));
    EXPECT_EQ(t1.shape(), Shape({3}));
    EXPECT_EQ(t1.elements(), 3U);
    
    // 2维
    float data2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t2 = Tensor::from_blob(data2, Shape{2, 2}, dtype(DataType::kFloat32));
    EXPECT_EQ(t2.shape(), Shape({2, 2}));
    EXPECT_EQ(t2.elements(), 4U);
    
    // 3维
    float data3[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto t3 = Tensor::from_blob(data3, Shape{2, 2, 2}, dtype(DataType::kFloat32));
    EXPECT_EQ(t3.shape(), Shape({2, 2, 2}));
    EXPECT_EQ(t3.elements(), 8U);
}

TEST_F(TensorCreateTest, FromBlobDifferentTypes)
{
    // 测试不同数据类型的from_blob
    Shape shape{2, 2};
    
    // float32
    float data_f32[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t_float32 = Tensor::from_blob(data_f32, shape, dtype(DataType::kFloat32));
    EXPECT_EQ(t_float32.dtype(), DataType::kFloat32);
    
    // int32
    int32_t data_i32[] = {1, 2, 3, 4};
    auto t_int32 = Tensor::from_blob(data_i32, shape, dtype(DataType::kInt32));
    EXPECT_EQ(t_int32.dtype(), DataType::kInt32);
    auto int_data = t_int32.to_vector<int32_t>();
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(int_data[i], data_i32[i]);
    }
}

TEST_F(TensorCreateTest, FromBlobDataIntegrity)
{
    // 测试from_blob的数据完整性
    float data[] = {1.1f, 2.2f, 3.3f, 4.4f};
    Shape shape{2, 2};
    auto t = Tensor::from_blob(data, shape, dtype(DataType::kFloat32));
    
    auto vec = t.to_vector<float>();
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(vec[i], data[i], origin::test::TestTolerance::kDefault);
    }
}

// ==================== 标量填充构造函数测试 ====================

TEST_F(TensorCreateTest, ScalarConstructorWithDataType)
{
    // 测试标量构造函数（指定数据类型）
    Shape shape{3, 3};
    float value = 5.0f;
    
    // 使用DataType参数
    auto t1 = Tensor(value, shape, DataType::kFloat32);
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.shape(), shape);
    auto expected1 = Tensor(std::vector<float>(9, value), shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(t1, expected1, origin::test::TestTolerance::kDefault);
    
    // 使用DataType::kFloat64
    auto t2 = Tensor(5.0, shape, DataType::kFloat64);
    EXPECT_EQ(t2.dtype(), DataType::kFloat64);
    EXPECT_EQ(t2.shape(), shape);
}

TEST_F(TensorCreateTest, ScalarConstructorWithTensorOptions)
{
    // 测试标量构造函数（指定TensorOptions）
    Shape shape{2, 2};
    float value = 3.14f;
    
    // 使用TensorOptions
    auto t1 = Tensor(value, shape, dtype(DataType::kFloat32));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.shape(), shape);
    auto expected1 = Tensor(std::vector<float>(4, value), shape, dtype(DataType::kFloat32));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(t1, expected1, origin::test::TestTolerance::kDefault);
    
    // 使用TensorOptions指定int32
    auto t2 = Tensor(42, shape, dtype(DataType::kInt32));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.shape(), shape);
    auto int_data = t2.to_vector<int32_t>();
    for (size_t i = 0; i < int_data.size(); ++i)
    {
        EXPECT_EQ(int_data[i], 42);
    }
}

TEST_F(TensorCreateTest, ScalarConstructorDifferentTypes)
{
    // 测试标量构造函数的不同数据类型
    Shape shape{2, 2};
    
    // float
    auto t_float = Tensor(2.5f, shape, DataType::kFloat32);
    EXPECT_EQ(t_float.dtype(), DataType::kFloat32);
    
    // double
    auto t_double = Tensor(2.5, shape, DataType::kFloat64);
    EXPECT_EQ(t_double.dtype(), DataType::kFloat64);
    
    // int32
    auto t_int32 = Tensor(42, shape, DataType::kInt32);
    EXPECT_EQ(t_int32.dtype(), DataType::kInt32);
    
    // int64
    auto t_int64 = Tensor(42L, shape, DataType::kInt64);
    EXPECT_EQ(t_int64.dtype(), DataType::kInt64);
}
