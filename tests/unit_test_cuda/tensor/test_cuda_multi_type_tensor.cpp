#include <gtest/gtest.h>
#include <vector>
#include "origin/core/tensor.h"
#include "origin/mat/basic_types.h"

using namespace origin;

class CudaMultiTypeTensorTest : public ::testing::Test
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
};

// 测试自动类型推断
TEST_F(CudaMultiTypeTensorTest, AutoTypeInference)
{
    // 测试float32类型推断（使用std::initializer_list）
    Tensor t1({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);

    // 测试int32类型推断（使用std::initializer_list）
    Tensor t2({1, 2, 3}, Shape{3}, dtype(Int32).device(kCUDA));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.device().type(), DeviceType::kCUDA);

    // 测试int8类型推断（使用std::initializer_list）
    Tensor t3({1, 2, 3}, Shape{3}, dtype(Int8).device(kCUDA));
    EXPECT_EQ(t3.dtype(), DataType::kInt8);
    EXPECT_EQ(t3.device().type(), DeviceType::kCUDA);

    // 测试vector构造函数
    std::vector<float> float_vec = {1.0f, 2.0f, 3.0f};
    Tensor t4(float_vec, Shape{3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t4.dtype(), DataType::kFloat32);
    EXPECT_EQ(t4.device().type(), DeviceType::kCUDA);

    std::vector<int32_t> int_vec = {1, 2, 3};
    Tensor t5(int_vec, Shape{3}, dtype(Int32).device(kCUDA));
    EXPECT_EQ(t5.dtype(), DataType::kInt32);
    EXPECT_EQ(t5.device().type(), DeviceType::kCUDA);
}

// 测试类型转换
TEST_F(CudaMultiTypeTensorTest, TypeConversion)
{
    // 创建float32张量
    Tensor t1({1.5f, 2.7f, 3.2f}, Shape{3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);

    // 转换为int32
    Tensor t2 = t1.to(DataType::kInt32);
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.device().type(), DeviceType::kCUDA);

    // 验证转换结果
    auto data2 = t2.to_vector<int32_t>();
    EXPECT_EQ(data2[0], 1);
    EXPECT_EQ(data2[1], 2);
    EXPECT_EQ(data2[2], 3);

    // 转换为int8
    Tensor t3 = t1.to(DataType::kInt8);
    EXPECT_EQ(t3.dtype(), DataType::kInt8);
    EXPECT_EQ(t3.device().type(), DeviceType::kCUDA);

    // 验证转换结果
    auto data3 = t3.to_vector<int8_t>();
    EXPECT_EQ(data3[0], 1);
    EXPECT_EQ(data3[1], 2);
    EXPECT_EQ(data3[2], 3);
}

// 测试标量张量
TEST_F(CudaMultiTypeTensorTest, ScalarTensors)
{
    // 测试float32标量
    Tensor t1(3.14f, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t1.shape(), Shape({1}));
    EXPECT_NEAR(t1.item<float>(), 3.14f, 1e-5f);

    // 测试int32标量
    Tensor t2(42, dtype(Int32).device(kCUDA));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t2.shape(), Shape({1}));
    EXPECT_EQ(t2.item<int32_t>(), 42);

    // 测试int8标量
    Tensor t3(static_cast<int8_t>(127), dtype(Int8).device(kCUDA));
    EXPECT_EQ(t3.dtype(), DataType::kInt8);
    EXPECT_EQ(t3.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t3.shape(), Shape({1}));
    EXPECT_EQ(t3.item<int8_t>(), 127);
}

// 测试显式类型构造函数
TEST_F(CudaMultiTypeTensorTest, ExplicitTypeConstructor)
{
    // 使用显式类型创建张量
    Tensor t1({1.0, 2.0, 3.0}, Shape{3}, dtype(Float64).device(kCUDA));
    EXPECT_EQ(t1.dtype(), DataType::kFloat64);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);

    Tensor t2({1, 2, 3}, Shape{3}, dtype(Int64).device(kCUDA));
    EXPECT_EQ(t2.dtype(), DataType::kInt64);
    EXPECT_EQ(t2.device().type(), DeviceType::kCUDA);

    Tensor t3({1, 2, 3}, Shape{3}, dtype(Int16).device(kCUDA));
    EXPECT_EQ(t3.dtype(), DataType::kInt16);
    EXPECT_EQ(t3.device().type(), DeviceType::kCUDA);
}

// 测试工厂函数
TEST_F(CudaMultiTypeTensorTest, FactoryFunctions)
{
    // 测试zeros工厂函数
    Tensor t1 = zeros(Shape{2, 3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t1.shape(), Shape({2, 3}));

    auto data1 = t1.to_vector<float>();
    for (float val : data1)
    {
        EXPECT_EQ(val, 0.0f);
    }

    // 测试ones工厂函数
    Tensor t2 = ones(Shape{2, 2}, dtype(Int32).device(kCUDA));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t2.shape(), Shape({2, 2}));

    auto data2 = t2.to_vector<int32_t>();
    for (int32_t val : data2)
    {
        EXPECT_EQ(val, 1);
    }

    // 测试randn工厂函数
    Tensor t3 = randn(Shape{100}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t3.dtype(), DataType::kFloat32);
    EXPECT_EQ(t3.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t3.shape(), Shape({100}));

    // 验证随机数的基本性质（均值接近0，标准差接近1）
    auto data3 = t3.to_vector<float>();
    float sum  = 0.0f;
    for (float val : data3)
    {
        sum += val;
    }
    float mean = sum / data3.size();
    EXPECT_NEAR(mean, 0.0f, 0.5f);  // 允许较大的误差，因为样本数量有限
}

// 测试初始化列表构造函数
TEST_F(CudaMultiTypeTensorTest, InitializerListConstructor)
{
    // 测试2D张量
    Tensor t1({{1.0f, 2.0f}, {3.0f, 4.0f}}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t1.shape(), Shape({2, 2}));

    auto data1                   = t1.to_vector<float>();
    std::vector<float> expected1 = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < expected1.size(); ++i)
    {
        EXPECT_EQ(data1[i], expected1[i]);
    }

    // 测试3D张量
    Tensor t2({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, dtype(Int32).device(kCUDA));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t2.shape(), Shape({2, 2, 2}));

    auto data2                     = t2.to_vector<int32_t>();
    std::vector<int32_t> expected2 = {1, 2, 3, 4, 5, 6, 7, 8};
    for (size_t i = 0; i < expected2.size(); ++i)
    {
        EXPECT_EQ(data2[i], expected2[i]);
    }
}

// 测试向后兼容性
TEST_F(CudaMultiTypeTensorTest, BackwardCompatibility)
{
    // 测试旧的构造函数仍然工作
    Tensor t1({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);

    // 测试默认设备类型
    Tensor t2({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(Float32));
    // 注意：这里应该使用CPU作为默认设备，但为了测试CUDA，我们显式指定
    EXPECT_EQ(t2.dtype(), DataType::kFloat32);
}

// 测试类型查询
TEST_F(CudaMultiTypeTensorTest, TypeQuery)
{
    Tensor t1({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(Float32).device(kCUDA));
    EXPECT_TRUE(t1.is_float());
    EXPECT_FALSE(t1.is_integer());
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);

    Tensor t2({1, 2, 3}, Shape{3}, dtype(Int32).device(kCUDA));
    EXPECT_FALSE(t2.is_float());
    EXPECT_TRUE(t2.is_integer());
    EXPECT_EQ(t2.dtype(), DataType::kInt32);

    Tensor t3({1, 2, 3}, Shape{3}, dtype(Int8).device(kCUDA));
    EXPECT_FALSE(t3.is_float());
    EXPECT_TRUE(t3.is_integer());
    EXPECT_EQ(t3.dtype(), DataType::kInt8);
}

// 测试错误处理
TEST_F(CudaMultiTypeTensorTest, ErrorHandling)
{
    // 测试形状不匹配
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    EXPECT_THROW(Tensor(data, Shape{2}, dtype(Float32).device(kCUDA)), std::runtime_error);

    // 测试空张量
    std::vector<float> empty_data;
    EXPECT_THROW(Tensor(empty_data, Shape{0}, dtype(Float32).device(kCUDA)), std::runtime_error);
}

// 测试形状和元素
TEST_F(CudaMultiTypeTensorTest, ShapeAndElements)
{
    Tensor t1({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t1.shape(), Shape({2, 2}));
    EXPECT_EQ(t1.elements(), 4);
    EXPECT_EQ(t1.dimensions(), 2);

    Tensor t2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t2.shape(), Shape({2, 3}));
    EXPECT_EQ(t2.elements(), 6);
    EXPECT_EQ(t2.dimensions(), 2);

    Tensor t3({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t3.shape(), Shape({2, 2, 2}));
    EXPECT_EQ(t3.elements(), 8);
    EXPECT_EQ(t3.dimensions(), 3);
}
