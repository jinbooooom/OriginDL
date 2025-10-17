#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
using namespace origin;

class CudaTensorCreateTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 检查CUDA可用性
        if (!cuda::is_cuda_available())
        {
            GTEST_SKIP() << "CUDA is not available on this system";
        }
    }

    void TearDown() override
    {
        // 清理CUDA资源
        cudaDeviceSynchronize();
    }

    // 辅助函数：比较两个浮点数是否相等（考虑浮点精度）
    bool isEqual(double a, double b, double tolerance = 1e-6) { return std::abs(a - b) < tolerance; }
};

// 从向量构造张量测试
TEST_F(CudaTensorCreateTest, ConstructorFromVector)
{
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};
    Tensor tensor(data, shape, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 4U);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);

    auto result_data = tensor.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], data[i], 1e-6);
    }
}

// 从初始化列表构造张量测试
TEST_F(CudaTensorCreateTest, ConstructorFromInitializerList)
{
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    Shape expected_shape{2, 2};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.elements(), 4U);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);

    auto result_data             = tensor.to_vector<float>();
    std::vector<data_t> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], 1e-6);
    }
}

// 从标量构造张量测试
TEST_F(CudaTensorCreateTest, ConstructorFromScalar)
{
    data_t scalar_value = 3.14;
    Shape shape{1};
    Tensor tensor(scalar_value, shape, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 1U);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);

    auto result_data = tensor.to_vector<float>();
    EXPECT_NEAR(result_data[0], scalar_value, 1e-6);
}

// 复制构造函数测试
TEST_F(CudaTensorCreateTest, CopyConstructor)
{
    Tensor original({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    Tensor copy(original);

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.elements(), original.elements());
    EXPECT_EQ(copy.device().type(), original.device().type());

    auto original_data = original.to_vector<float>();
    auto copy_data     = copy.to_vector<float>();

    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(copy_data[i], original_data[i], 1e-6);
    }
}

// 移动构造函数测试
TEST_F(CudaTensorCreateTest, MoveConstructor)
{
    Tensor original({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto original_data = original.to_vector<float>();

    Tensor moved(std::move(original));

    EXPECT_EQ(moved.shape(), Shape({2, 2}));
    EXPECT_EQ(moved.elements(), 4U);
    EXPECT_EQ(moved.device().type(), DeviceType::kCUDA);

    auto moved_data = moved.to_vector<float>();
    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(moved_data[i], original_data[i], 1e-6);
    }

    // 验证原张量已被移动
    // 注意：访问已移动的对象是未定义行为，这里只是演示
    // 在实际代码中应该避免这样做
}

// 赋值操作符测试
TEST_F(CudaTensorCreateTest, AssignmentOperators)
{
    Tensor tensor1({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    Tensor tensor2({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    // 复制赋值
    tensor2 = tensor1;

    EXPECT_EQ(tensor2.shape(), tensor1.shape());
    EXPECT_EQ(tensor2.elements(), tensor1.elements());
    EXPECT_EQ(tensor2.device().type(), tensor1.device().type());

    auto data1 = tensor1.to_vector<float>();
    auto data2 = tensor2.to_vector<float>();

    for (size_t i = 0; i < data1.size(); ++i)
    {
        EXPECT_NEAR(data2[i], data1[i], 1e-6);
    }

    // 移动赋值
    Tensor tensor3({9.0f, 10.0f, 11.0f, 12.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto tensor3_data = tensor3.to_vector<float>();

    tensor2 = std::move(tensor3);

    EXPECT_EQ(tensor2.shape(), Shape({2, 2}));
    EXPECT_EQ(tensor2.elements(), 4U);
    EXPECT_EQ(tensor2.device().type(), DeviceType::kCUDA);

    auto data2_after_move = tensor2.to_vector<float>();
    for (size_t i = 0; i < tensor3_data.size(); ++i)
    {
        EXPECT_NEAR(data2_after_move[i], tensor3_data[i], 1e-6);
    }
}

// 工厂方法测试
TEST_F(CudaTensorCreateTest, FactoryMethods)
{
    // 测试zeros工厂方法
    Tensor zeros_tensor = zeros(Shape{3, 3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(zeros_tensor.shape(), Shape({3, 3}));
    EXPECT_EQ(zeros_tensor.elements(), 9U);
    EXPECT_EQ(zeros_tensor.device().type(), DeviceType::kCUDA);

    auto zeros_data = zeros_tensor.to_vector<float>();
    for (float val : zeros_data)
    {
        EXPECT_EQ(val, 0.0f);
    }

    // 测试ones工厂方法
    Tensor ones_tensor = ones(Shape{2, 4}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(ones_tensor.shape(), Shape({2, 4}));
    EXPECT_EQ(ones_tensor.elements(), 8U);
    EXPECT_EQ(ones_tensor.device().type(), DeviceType::kCUDA);

    auto ones_data = ones_tensor.to_vector<float>();
    for (float val : ones_data)
    {
        EXPECT_EQ(val, 1.0f);
    }
}

// randn工厂方法测试
TEST_F(CudaTensorCreateTest, RandnFactory)
{
    Tensor randn_tensor = randn(Shape{100}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(randn_tensor.shape(), Shape({100}));
    EXPECT_EQ(randn_tensor.elements(), 100U);
    EXPECT_EQ(randn_tensor.device().type(), DeviceType::kCUDA);

    auto randn_data = randn_tensor.to_vector<float>();

    // 验证随机数的基本性质（均值接近0，标准差接近1）
    float sum = 0.0f;
    for (float val : randn_data)
    {
        sum += val;
    }
    float mean = sum / randn_data.size();
    EXPECT_NEAR(mean, 0.0f, 0.5f);  // 允许较大的误差，因为样本数量有限

    // 验证所有值都是有限的
    for (float val : randn_data)
    {
        EXPECT_TRUE(std::isfinite(val));
    }
}

// 形状验证测试
TEST_F(CudaTensorCreateTest, ShapeValidation)
{
    // 测试有效形状
    EXPECT_NO_THROW(Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA)));
    EXPECT_NO_THROW(Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA)));
    EXPECT_NO_THROW(
        Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2}, dtype(Float32).device(kCUDA)));

    // 测试无效形状（元素数量不匹配）
    EXPECT_THROW(Tensor({1.0f, 2.0f, 3.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA)), std::runtime_error);
    EXPECT_THROW(Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA)), std::runtime_error);
}

// 空张量测试
TEST_F(CudaTensorCreateTest, EmptyTensor)
{
    // 测试空张量
    Tensor empty_tensor({}, Shape{0}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(empty_tensor.shape(), Shape({0}));
    EXPECT_EQ(empty_tensor.elements(), 0U);
    EXPECT_EQ(empty_tensor.device().type(), DeviceType::kCUDA);

    auto empty_data = empty_tensor.to_vector<float>();
    EXPECT_TRUE(empty_data.empty());
}

// 标量张量测试
TEST_F(CudaTensorCreateTest, ScalarTensor)
{
    // 测试标量张量
    Tensor scalar_tensor(42.0f, dtype(Float32).device(kCUDA));
    EXPECT_EQ(scalar_tensor.shape(), Shape({1}));
    EXPECT_EQ(scalar_tensor.elements(), 1U);
    EXPECT_EQ(scalar_tensor.device().type(), DeviceType::kCUDA);

    auto scalar_data = scalar_tensor.to_vector<float>();
    EXPECT_EQ(scalar_data.size(), 1U);
    EXPECT_NEAR(scalar_data[0], 42.0f, 1e-6);
}

// 大张量测试
TEST_F(CudaTensorCreateTest, LargeTensor)
{
    const size_t size = 10000;
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    Tensor large_tensor(data, Shape{size}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(large_tensor.shape(), Shape({size}));
    EXPECT_EQ(large_tensor.elements(), size);
    EXPECT_EQ(large_tensor.device().type(), DeviceType::kCUDA);

    auto large_data = large_tensor.to_vector<float>();
    EXPECT_EQ(large_data.size(), size);

    // 验证前100个元素
    for (size_t i = 0; i < std::min(size, size_t(100)); ++i)
    {
        EXPECT_NEAR(large_data[i], static_cast<float>(i), 1e-6);
    }
}

// 一维张量测试
TEST_F(CudaTensorCreateTest, OneDimensionalTensor)
{
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(tensor.shape(), Shape({5}));
    EXPECT_EQ(tensor.elements(), 5U);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);

    auto data                   = tensor.to_vector<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data[i], expected[i], 1e-6);
    }
}

// 三维张量测试
TEST_F(CudaTensorCreateTest, ThreeDimensionalTensor)
{
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(tensor.shape(), Shape({2, 2, 2}));
    EXPECT_EQ(tensor.elements(), 8U);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);

    auto data                   = tensor.to_vector<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data[i], expected[i], 1e-6);
    }
}

// 数据完整性测试
TEST_F(CudaTensorCreateTest, DataIntegrity)
{
    std::vector<float> original_data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f};
    Tensor tensor(original_data, Shape{2, 3}, dtype(Float32).device(kCUDA));

    auto retrieved_data = tensor.to_vector<float>();
    EXPECT_EQ(retrieved_data.size(), original_data.size());

    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(retrieved_data[i], original_data[i], 1e-6);
    }
}

// 内存管理测试
TEST_F(CudaTensorCreateTest, MemoryManagement)
{
    // 测试多个张量的内存管理
    std::vector<Tensor> tensors;
    for (int i = 0; i < 10; ++i)
    {
        std::vector<float> data = {static_cast<float>(i), static_cast<float>(i + 1)};
        tensors.emplace_back(data, Shape{2}, dtype(Float32).device(kCUDA));
    }

    // 验证所有张量都正确创建
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        EXPECT_EQ(tensors[i].shape(), Shape({2}));
        EXPECT_EQ(tensors[i].device().type(), DeviceType::kCUDA);
        auto data = tensors[i].to_vector<float>();
        EXPECT_NEAR(data[0], static_cast<float>(i), 1e-6);
        EXPECT_NEAR(data[1], static_cast<float>(i + 1), 1e-6);
    }
}

// 张量内存生命周期测试
TEST_F(CudaTensorCreateTest, TensorMemoryLifecycle)
{
    // 测试张量在作用域结束后的内存管理
    {
        Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
        EXPECT_EQ(tensor.elements(), 4U);
        EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);
    }
    // 张量应该在这里被销毁，内存应该被释放

    // 创建新的张量，验证内存可以重新使用
    Tensor new_tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(new_tensor.elements(), 4U);
    EXPECT_EQ(new_tensor.device().type(), DeviceType::kCUDA);

    auto data                   = new_tensor.to_vector<float>();
    std::vector<float> expected = {5.0f, 6.0f, 7.0f, 8.0f};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data[i], expected[i], 1e-6);
    }
}

// 不同数据类型创建测试
TEST_F(CudaTensorCreateTest, DifferentDataTypeCreation)
{
    // 测试Float32
    Tensor float32_tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(float32_tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(float32_tensor.device().type(), DeviceType::kCUDA);

    // 测试Int32
    Tensor int32_tensor({1, 2, 3}, Shape{3}, dtype(Int32).device(kCUDA));
    EXPECT_EQ(int32_tensor.dtype(), DataType::kInt32);
    EXPECT_EQ(int32_tensor.device().type(), DeviceType::kCUDA);

    // 测试Int8
    Tensor int8_tensor({1, 2, 3}, Shape{3}, dtype(Int8).device(kCUDA));
    EXPECT_EQ(int8_tensor.dtype(), DataType::kInt8);
    EXPECT_EQ(int8_tensor.device().type(), DeviceType::kCUDA);

    // 验证数据正确性
    auto float32_data = float32_tensor.to_vector<float>();
    auto int32_data   = int32_tensor.to_vector<int32_t>();
    auto int8_data    = int8_tensor.to_vector<int8_t>();

    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(float32_data[i], static_cast<float>(i + 1), 1e-6);
        EXPECT_EQ(int32_data[i], i + 1);
        EXPECT_EQ(int8_data[i], static_cast<int8_t>(i + 1));
    }
}
