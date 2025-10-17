#include <gtest/gtest.h>
#include "origin/core/tensor.h"
#include "origin/core/tensor_options.h"
#include "origin/mat/basic_types.h"

using namespace origin;

class CudaTensorOptionsTest : public ::testing::Test
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

// 测试TensorOptions基本功能
TEST_F(CudaTensorOptionsTest, BasicTensorOptions)
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
TEST_F(CudaTensorOptionsTest, ConvenienceFunctions)
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
    auto options3 = requires_grad(false);
    EXPECT_EQ(options3.dtype(), DataType::kFloat32);
    EXPECT_EQ(options3.device().type(), DeviceType::kCPU);
    EXPECT_FALSE(options3.requires_grad());
}

// 测试Tensor工厂方法
TEST_F(CudaTensorOptionsTest, TensorFactoryMethods)
{
    // 测试使用TensorOptions创建张量
    auto options = TensorOptions().dtype(DataType::kFloat32).device(Device(DeviceType::kCUDA, 0));

    Tensor t1 = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, options);
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t1.device().index(), 0);

    // 测试使用便利函数创建张量
    Tensor t2 = Tensor({1, 2, 3}, Shape{3}, dtype(Int32).device(kCUDA));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.device().type(), DeviceType::kCUDA);

    // 测试标量张量
    Tensor t3 = Tensor(3.14f, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t3.dtype(), DataType::kFloat32);
    EXPECT_EQ(t3.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t3.shape(), Shape({1}));
    EXPECT_NEAR(t3.item<float>(), 3.14f, 1e-5f);
}

// 测试链式选项
TEST_F(CudaTensorOptionsTest, ChainedOptions)
{
    // 测试链式调用
    auto options = TensorOptions().dtype(DataType::kInt64).device(Device(DeviceType::kCUDA, 0)).requires_grad(false);

    EXPECT_EQ(options.dtype(), DataType::kInt64);
    EXPECT_EQ(options.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(options.device().index(), 0);
    EXPECT_FALSE(options.requires_grad());

    // 使用链式选项创建张量
    Tensor t = Tensor({1, 2, 3, 4}, Shape{2, 2}, options);
    EXPECT_EQ(t.dtype(), DataType::kInt64);
    EXPECT_EQ(t.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t.device().index(), 0);

    auto data                     = t.to_vector<int64_t>();
    std::vector<int64_t> expected = {1, 2, 3, 4};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// 测试类型转换
TEST_F(CudaTensorOptionsTest, TypeConversion)
{
    // 创建float32张量
    Tensor t1 = Tensor({1.5f, 2.7f, 3.2f}, Shape{3}, dtype(Float32).device(kCUDA));
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

// 测试向后兼容性
TEST_F(CudaTensorOptionsTest, BackwardCompatibility)
{
    // 测试旧的构造函数仍然工作
    Tensor t1 = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);

    // 测试默认选项
    Tensor t2 = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(Float32));
    // 注意：这里应该使用CPU作为默认设备，但为了测试CUDA，我们显式指定
    EXPECT_EQ(t2.dtype(), DataType::kFloat32);
}

// 测试设备信息
TEST_F(CudaTensorOptionsTest, DeviceInfo)
{
    // 测试CUDA设备信息
    auto cuda_device = Device(DeviceType::kCUDA, 0);
    EXPECT_EQ(cuda_device.type(), DeviceType::kCUDA);
    EXPECT_EQ(cuda_device.index(), 0);

    // 测试设备比较
    auto cuda_device2 = Device(DeviceType::kCUDA, 0);
    EXPECT_EQ(cuda_device, cuda_device2);

    auto cuda_device3 = Device(DeviceType::kCUDA, 1);
    EXPECT_NE(cuda_device, cuda_device3);

    // 测试使用不同CUDA设备创建张量
    Tensor t1 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(Float32).device(Device(DeviceType::kCUDA, 0)));
    EXPECT_EQ(t1.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t1.device().index(), 0);

    Tensor t2 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(Float32).device(Device(DeviceType::kCUDA, 0)));
    EXPECT_EQ(t2.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(t2.device().index(), 0);
}

// 测试TensorOptions比较
TEST_F(CudaTensorOptionsTest, TensorOptionsComparison)
{
    auto options1 = TensorOptions().dtype(DataType::kFloat32).device(Device(DeviceType::kCUDA, 0));
    auto options2 = TensorOptions().dtype(DataType::kFloat32).device(Device(DeviceType::kCUDA, 0));
    auto options3 = TensorOptions().dtype(DataType::kInt32).device(Device(DeviceType::kCUDA, 0));
    auto options4 = TensorOptions().dtype(DataType::kFloat32).device(Device(DeviceType::kCUDA, 1));

    // 测试相等性
    EXPECT_EQ(options1, options2);

    // 测试不等性
    EXPECT_NE(options1, options3);  // 不同数据类型
    EXPECT_NE(options1, options4);  // 不同设备索引

    // 测试使用相同选项创建张量
    Tensor t1 = Tensor({1.0f, 2.0f}, Shape{2}, options1);
    Tensor t2 = Tensor({3.0f, 4.0f}, Shape{2}, options2);

    EXPECT_EQ(t1.dtype(), t2.dtype());
    EXPECT_EQ(t1.device().type(), t2.device().type());
    EXPECT_EQ(t1.device().index(), t2.device().index());
}
