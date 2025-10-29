#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA张量基础功能测试类
 * @details 测试CUDA张量的创建、转换、属性等基础功能
 */
class CudaTensorTest : public ::testing::Test
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

    // 精度容忍常量
    static constexpr double kFloatTolerance = 1e-5;
};

// ============================================================================
// 张量创建测试
// ============================================================================

TEST_F(CudaTensorTest, TensorCreation)
{
    // 测试CUDA张量创建
    auto tensor = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.shape(), Shape({2, 2}));
    EXPECT_EQ(tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);

    auto data                   = tensor.to_vector<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaTensorTest, DifferentDataTypes)
{
    // 测试不同数据类型的CUDA张量创建
    auto float_tensor = Tensor(std::vector<float>{1.5f, 2.5f}, Shape{2}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(float_tensor.dtype(), DataType::kFloat32);

    auto double_tensor = Tensor(std::vector<double>{1.5, 2.5}, Shape{2}, dtype(Float64).device(kCUDA));
    EXPECT_EQ(double_tensor.dtype(), DataType::kFloat64);

    auto int_tensor = Tensor(std::vector<int32_t>{1, 2}, Shape{2}, dtype(Int32).device(kCUDA));
    EXPECT_EQ(int_tensor.dtype(), DataType::kInt32);
}

TEST_F(CudaTensorTest, DifferentShapes)
{
    // 测试不同形状的CUDA张量创建
    auto tensor_1d = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f}, Shape{3}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(tensor_1d.shape(), Shape({3}));

    auto tensor_2d = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    EXPECT_EQ(tensor_2d.shape(), Shape({2, 2}));

    auto tensor_3d = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                            dtype(Float32).device(kCUDA));
    EXPECT_EQ(tensor_3d.shape(), Shape({2, 2, 2}));
}

TEST_F(CudaTensorTest, SingleElementTensor)
{
    // 测试单元素张量
    auto tensor = Tensor(std::vector<float>{42.0f}, Shape{1}, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.shape(), Shape({1}));
    EXPECT_EQ(tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);

    auto data = tensor.to_vector<float>();
    EXPECT_NEAR(data[0], 42.0f, kFloatTolerance);
}

TEST_F(CudaTensorTest, ZeroTensor)
{
    // 测试零值张量
    auto tensor = Tensor(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.shape(), Shape({2, 2}));
    auto data = tensor.to_vector<float>();

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(data[i], 0.0f, kFloatTolerance);
    }
}

// ============================================================================
// 张量属性测试
// ============================================================================

TEST_F(CudaTensorTest, DeviceConsistency)
{
    // 测试设备一致性
    auto tensor = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor.shape(), Shape({2, 2}));
}

TEST_F(CudaTensorTest, ShapeValidation)
{
    // 测试形状验证
    auto tensor =
        Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.shape().elements(), 6U);
    EXPECT_EQ(tensor.shape().ndims(), 2U);
    EXPECT_EQ(tensor.shape()[0], 2U);
    EXPECT_EQ(tensor.shape()[1], 3U);
}

// ============================================================================
// 数据转换测试
// ============================================================================

TEST_F(CudaTensorTest, DataRetrieval)
{
    // 测试数据检索
    std::vector<float> original_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto tensor                      = Tensor(original_data, Shape{2, 2});

    auto retrieved_data = tensor.to_vector<float>();

    EXPECT_EQ(retrieved_data.size(), original_data.size());
    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(retrieved_data[i], original_data[i], kFloatTolerance);
    }
}

TEST_F(CudaTensorTest, DataIntegrity)
{
    // 测试数据完整性
    std::vector<float> test_data;
    for (int i = 0; i < 100; ++i)
    {
        test_data.push_back(static_cast<float>(i));
    }

    auto tensor         = Tensor(test_data, Shape{10, 10});
    auto retrieved_data = tensor.to_vector<float>();

    EXPECT_EQ(retrieved_data.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); ++i)
    {
        EXPECT_NEAR(retrieved_data[i], test_data[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaTensorTest, LargeTensor)
{
    // 测试大张量创建
    const size_t size = 1000;
    std::vector<float> large_data(size);
    for (size_t i = 0; i < size; ++i)
    {
        large_data[i] = static_cast<float>(i);
    }

    auto tensor = Tensor(large_data, Shape{size}, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.shape(), Shape({size}));
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);

    // 只检查前100个元素以节省时间
    auto retrieved_data = tensor.to_vector<float>();
    for (size_t i = 0; i < std::min(size, size_t(100)); ++i)
    {
        EXPECT_NEAR(retrieved_data[i], large_data[i], kFloatTolerance);
    }
}

TEST_F(CudaTensorTest, NegativeValues)
{
    // 测试负值张量
    auto tensor = Tensor(std::vector<float>{-1.0f, -2.0f, -3.0f, -4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    EXPECT_EQ(tensor.shape(), Shape({2, 2}));
    auto data                   = tensor.to_vector<float>();
    std::vector<float> expected = {-1.0f, -2.0f, -3.0f, -4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// PyTorch风格构造测试
// ============================================================================

TEST_F(CudaTensorTest, PyTorchStyleConstruction)
{
    // 测试类似PyTorch的构造方式：dtype(Float32).device("cuda")
    auto tensor1 = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device("cuda"));

    EXPECT_EQ(tensor1.shape(), Shape({2, 2}));
    EXPECT_EQ(tensor1.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor1.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor1.device().index(), 0);  // 默认GPU 0

    auto data1                   = tensor1.to_vector<float>();
    std::vector<float> expected1 = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < expected1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], expected1[i], kFloatTolerance);
    }
}

TEST_F(CudaTensorTest, PyTorchStyleStringDtypeAndDevice)
{
    // 测试字符串形式的dtype和device组合：dtype("float32").device("cuda")
    auto tensor1 = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype("float32").device("cuda"));

    EXPECT_EQ(tensor1.shape(), Shape({2, 2}));
    EXPECT_EQ(tensor1.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor1.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor1.device().index(), 0);

    auto data1                   = tensor1.to_vector<float>();
    std::vector<float> expected1 = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < expected1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], expected1[i], kFloatTolerance);
    }

    // 测试不同数据类型的字符串形式
    auto tensor2 = Tensor(std::vector<int32_t>{1, 2, 3, 4}, Shape{2, 2}, dtype("int32").device("cuda:0"));
    EXPECT_EQ(tensor2.dtype(), DataType::kInt32);
    EXPECT_EQ(tensor2.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor2.device().index(), 0);

    auto tensor3 = Tensor(std::vector<double>{1.0, 2.0, 3.0, 4.0}, Shape{2, 2}, dtype("float64").device("cuda"));
    EXPECT_EQ(tensor3.dtype(), DataType::kFloat64);
    EXPECT_EQ(tensor3.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor3.device().index(), 0);

    auto tensor4 = Tensor(std::vector<int8_t>{1, 2, 3, 4}, Shape{2, 2}, dtype("int8").device("cuda"));
    EXPECT_EQ(tensor4.dtype(), DataType::kInt8);
    EXPECT_EQ(tensor4.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor4.device().index(), 0);
}

TEST_F(CudaTensorTest, PyTorchStyleConstructionWithDeviceIndex)
{
    // 测试指定GPU设备索引：dtype(Float32).device("cuda:0")
    auto tensor1 = Tensor(std::vector<float>{1.0f, 2.0f}, Shape{2}, dtype(Float32).device("cuda:0"));

    EXPECT_EQ(tensor1.shape(), Shape({2}));
    EXPECT_EQ(tensor1.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor1.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor1.device().index(), 0);

    // 测试不同数据类型
    auto tensor2 = Tensor(std::vector<int32_t>{1, 2, 3, 4}, Shape{2, 2}, dtype(Int32).device("cuda"));
    EXPECT_EQ(tensor2.dtype(), DataType::kInt32);
    EXPECT_EQ(tensor2.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor2.device().index(), 0);
}

TEST_F(CudaTensorTest, PyTorchStyleFactoryMethods)
{
    // 测试工厂方法的PyTorch风格构造
    Shape shape{3, 3};

    // zeros with PyTorch style - 使用字符串形式
    auto zeros_tensor = Tensor::zeros(shape, dtype("float32").device("cuda"));
    EXPECT_EQ(zeros_tensor.shape(), shape);
    EXPECT_EQ(zeros_tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(zeros_tensor.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(zeros_tensor.device().index(), 0);

    auto zeros_data = zeros_tensor.to_vector<float>();
    for (size_t i = 0; i < zeros_data.size(); ++i)
    {
        EXPECT_NEAR(zeros_data[i], 0.0f, kFloatTolerance);
    }

    // ones with PyTorch style - 使用字符串形式
    auto ones_tensor = Tensor::ones(shape, dtype("float32").device("cuda:0"));
    EXPECT_EQ(ones_tensor.shape(), shape);
    EXPECT_EQ(ones_tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(ones_tensor.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(ones_tensor.device().index(), 0);

    auto ones_data = ones_tensor.to_vector<float>();
    for (size_t i = 0; i < ones_data.size(); ++i)
    {
        EXPECT_NEAR(ones_data[i], 1.0f, kFloatTolerance);
    }

    // randn with PyTorch style - 使用字符串形式
    auto randn_tensor = Tensor::randn(shape, dtype("float32").device("cuda"));
    EXPECT_EQ(randn_tensor.shape(), shape);
    EXPECT_EQ(randn_tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(randn_tensor.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(randn_tensor.device().index(), 0);
}

TEST_F(CudaTensorTest, PyTorchStyleInitializerList)
{
    // 测试初始化列表的PyTorch风格构造 - 使用字符串形式
    auto tensor = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype("float32").device("cuda"));

    EXPECT_EQ(tensor.shape(), Shape({2, 2}));
    EXPECT_EQ(tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor.device().index(), 0);

    auto data                   = tensor.to_vector<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaTensorTest, PyTorchStyleScalarConstruction)
{
    // 测试标量构造的PyTorch风格 - 使用字符串形式
    auto tensor = Tensor(42.0f, Shape{2, 2}, dtype("float32").device("cuda"));

    EXPECT_EQ(tensor.shape(), Shape({2, 2}));
    EXPECT_EQ(tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor.device().type(), DeviceType::kCUDA);
    EXPECT_EQ(tensor.device().index(), 0);

    auto data = tensor.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(data[i], 42.0f, kFloatTolerance);
    }
}