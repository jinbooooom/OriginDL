#include <gtest/gtest.h>
#include "origin/core/tensor.h"
#include "origin/core/tensor_options.h"
#include "origin/mat/basic_types.h"

using namespace origin;

class TensorOptionsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 测试设置
    }

    void TearDown() override
    {
        // 测试清理
    }
};

// 测试TensorOptions基本功能
TEST_F(TensorOptionsTest, BasicTensorOptions)
{
    // 测试默认构造
    TensorOptions options1;
    EXPECT_EQ(options1.dtype(), DataType::kFloat32);
    EXPECT_EQ(options1.device().type(), DeviceType::kCPU);
    EXPECT_TRUE(options1.requires_grad());  // 当前origindl默认requires_grad=true

    // 测试链式调用
    auto options2 = TensorOptions().dtype(DataType::kInt32).device(Device(DeviceType::kCUDA, 0)).requires_grad(true);

    EXPECT_EQ(options2.dtype(), DataType::kInt32);
    EXPECT_EQ(options2.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(options2.device().index(), 0);
    EXPECT_TRUE(options2.requires_grad());
}

// 测试便利函数
TEST_F(TensorOptionsTest, ConvenienceFunctions)
{
    // 测试dtype便利函数
    auto options1 = dtype(DataType::kInt32);
    EXPECT_EQ(options1.dtype(), DataType::kInt32);
    EXPECT_EQ(options1.device().type(), DeviceType::kCPU);

    // 测试device便利函数
    auto options2 = device(Device(DeviceType::kCUDA, 1));
    EXPECT_EQ(options2.dtype(), DataType::kFloat32);
    EXPECT_EQ(options2.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(options2.device().index(), 1);

    // 测试requires_grad便利函数
    auto options3 = requires_grad(true);
    EXPECT_EQ(options3.dtype(), DataType::kFloat32);
    EXPECT_TRUE(options3.requires_grad());
}

// 测试Tensor工厂方法支持TensorOptions
TEST_F(TensorOptionsTest, TensorFactoryMethods)
{
    Shape shape{2, 3};

    // 测试zeros
    auto t1 = Tensor::zeros(shape, dtype(DataType::kInt32));
    EXPECT_EQ(t1.dtype(), DataType::kInt32);
    EXPECT_EQ(t1.shape(), shape);

    // 测试ones
    auto t2 = Tensor::ones(shape, dtype(DataType::kFloat32));
    EXPECT_EQ(t2.dtype(), DataType::kFloat32);
    EXPECT_EQ(t2.shape(), shape);

    // 测试randn
    auto t3 = Tensor::randn(shape, dtype(DataType::kFloat32));
    EXPECT_EQ(t3.dtype(), DataType::kFloat32);
    EXPECT_EQ(t3.shape(), shape);

    // 测试full
    auto t4 = Tensor::full(shape, 5.0, dtype(DataType::kInt32));
    EXPECT_EQ(t4.dtype(), DataType::kInt32);
    EXPECT_EQ(t4.shape(), shape);
}

// 测试链式调用
TEST_F(TensorOptionsTest, ChainedOptions)
{
    Shape shape{2, 2};

    // 测试链式调用创建张量
    auto options = dtype(DataType::kInt32).device(Device(DeviceType::kCPU)).requires_grad(true);

    auto t = Tensor::zeros(shape, options);
    EXPECT_EQ(t.dtype(), DataType::kInt32);
    EXPECT_EQ(t.shape(), shape);
}

// 测试类型转换
TEST_F(TensorOptionsTest, TypeConversion)
{
    Shape shape{2, 2};

    // 创建float32张量
    auto t1 = Tensor::zeros(shape, dtype(DataType::kFloat32));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);

    // 转换为int32
    auto t2 = t1.to(dtype(DataType::kInt32));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.shape(), shape);
}

// 测试向后兼容性
TEST_F(TensorOptionsTest, BackwardCompatibility)
{
    Shape shape{2, 2};

    // 测试TensorOptions方式
    auto t1 = Tensor::zeros(shape, dtype(DataType::kFloat32));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);

    auto t2 = Tensor::ones(shape, dtype(DataType::kInt32));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);

    auto t3 = Tensor::randn(shape, dtype(DataType::kFloat32));
    EXPECT_EQ(t3.dtype(), DataType::kFloat32);
}

// 测试设备信息
TEST_F(TensorOptionsTest, DeviceInfo)
{
    Device cpu_device(DeviceType::kCPU);
    Device cuda_device(DeviceType::kCUDA, 0);

    EXPECT_EQ(cpu_device.type(), DeviceType::kCPU);
    EXPECT_EQ(cuda_device.type(), DeviceType::kCUDA);
    EXPECT_EQ(cuda_device.index(), 0);

    EXPECT_EQ(cpu_device.to_string(), "cpu");
    EXPECT_EQ(cuda_device.to_string(), "cuda:0");
}

// 测试TensorOptions比较
TEST_F(TensorOptionsTest, TensorOptionsComparison)
{
    auto options1 = dtype(DataType::kFloat32).device(Device(DeviceType::kCPU));
    auto options2 = dtype(DataType::kFloat32).device(Device(DeviceType::kCPU));
    auto options3 = dtype(DataType::kInt32).device(Device(DeviceType::kCPU));

    EXPECT_EQ(options1, options2);
    EXPECT_NE(options1, options3);
}
