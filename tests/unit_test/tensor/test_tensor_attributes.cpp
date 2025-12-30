#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;

/**
 * @brief 张量属性测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 *          无GPU环境只运行CPU测试，有GPU环境运行CPU+CUDA测试
 */
class TensorAttributesTest : public origin::test::OperatorTestBase
{};

// ==================== shape() 测试 ====================

TEST_P(TensorAttributesTest, ShapeBasic)
{
    // 测试基本形状获取
    Shape shape{2, 3};
    auto t = Tensor::ones(shape, dtype(DataType::kFloat32).device(deviceType()));

    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.shape().size(), 2U);
    EXPECT_EQ(t.shape()[0], 2U);
    EXPECT_EQ(t.shape()[1], 3U);
}

TEST_P(TensorAttributesTest, ShapeDifferentDimensions)
{
    // 测试不同维度的形状
    // 1维
    auto t1 = Tensor::ones(Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t1.shape(), Shape({5}));

    // 2维
    auto t2 = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t2.shape(), Shape({2, 3}));

    // 3维
    auto t3 = Tensor::ones(Shape{2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t3.shape(), Shape({2, 3, 4}));

    // 4维
    auto t4 = Tensor::ones(Shape{2, 3, 4, 5}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t4.shape(), Shape({2, 3, 4, 5}));
}

// ==================== ndim() 测试 ====================

TEST_P(TensorAttributesTest, NdimBasic)
{
    // 测试基本维度数获取
    auto t = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t.ndim(), 2U);
}

TEST_P(TensorAttributesTest, NdimDifferentDimensions)
{
    // 测试0维（标量）
    auto t0 = Tensor::full(Shape{1}, 1.0f, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t0.ndim(), 1U);  // 形状为{1}，维度为1

    // 测试1维
    auto t1 = Tensor::ones(Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t1.ndim(), 1U);

    // 测试2维
    auto t2 = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t2.ndim(), 2U);

    // 测试3维
    auto t3 = Tensor::ones(Shape{2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t3.ndim(), 3U);
}

// ==================== elements() / numel() 测试 ====================

TEST_P(TensorAttributesTest, ElementsBasic)
{
    // 测试基本元素数获取
    auto t = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t.elements(), 6U);
    EXPECT_EQ(t.numel(), 6U);
    EXPECT_EQ(t.elements(), t.numel());
}

TEST_P(TensorAttributesTest, ElementsDifferentShapes)
{
    // 测试不同形状的元素数
    // 1维
    auto t1 = Tensor::ones(Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t1.elements(), 5U);
    EXPECT_EQ(t1.numel(), 5U);

    // 2维
    auto t2 = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t2.elements(), 6U);
    EXPECT_EQ(t2.numel(), 6U);

    // 3维
    auto t3 = Tensor::ones(Shape{2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t3.elements(), 24U);
    EXPECT_EQ(t3.numel(), 24U);

    // 标量
    auto t0 = Tensor::full(Shape{1}, 1.0f, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t0.elements(), 1U);
    EXPECT_EQ(t0.numel(), 1U);
}

// ==================== dtype() 测试 ====================

TEST_P(TensorAttributesTest, DtypeBasic)
{
    // 测试基本数据类型获取
    auto t = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t.dtype(), DataType::kFloat32);
}

TEST_P(TensorAttributesTest, DtypeDifferentTypes)
{
    // 测试不同数据类型
    auto t_float32 = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t_float32.dtype(), DataType::kFloat32);

    auto t_float64 = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));
    EXPECT_EQ(t_float64.dtype(), DataType::kFloat64);

    auto t_int32 = Tensor::ones(Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    EXPECT_EQ(t_int32.dtype(), DataType::kInt32);

    auto t_int64 = Tensor::ones(Shape{2, 2}, dtype(DataType::kInt64).device(deviceType()));
    EXPECT_EQ(t_int64.dtype(), DataType::kInt64);
}

// ==================== device() 测试 ====================

TEST_P(TensorAttributesTest, DeviceBasic)
{
    // 测试基本设备获取
    auto t = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t.device().type(), deviceType());

    if (deviceType() == DeviceType::kCUDA)
    {
        EXPECT_EQ(t.device().index(), 0);
    }
}

TEST_P(TensorAttributesTest, DeviceCPU)
{
    // 测试CPU设备
    auto t = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
    EXPECT_EQ(t.device().type(), DeviceType::kCPU);
}

TEST_P(TensorAttributesTest, DeviceCUDA)
{
    // 测试CUDA设备（如果可用）
    if (deviceType() == DeviceType::kCUDA)
    {
        auto t = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));
        EXPECT_EQ(t.device().type(), DeviceType::kCUDA);
        EXPECT_EQ(t.device().index(), 0);
    }
    else
    {
        GTEST_SKIP() << "CUDA is not available";
    }
}

// ==================== element_size() 测试 ====================

TEST_P(TensorAttributesTest, ElementSizeBasic)
{
    // 测试基本元素大小获取
    auto t = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t.element_size(), 4U);  // float32 占4字节
}

TEST_P(TensorAttributesTest, ElementSizeDifferentTypes)
{
    // 测试不同数据类型的元素大小
    auto t_float32 = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t_float32.element_size(), 4U);  // float32 = 4字节

    auto t_float64 = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));
    EXPECT_EQ(t_float64.element_size(), 8U);  // float64 = 8字节

    auto t_int32 = Tensor::ones(Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    EXPECT_EQ(t_int32.element_size(), 4U);  // int32 = 4字节

    auto t_int64 = Tensor::ones(Shape{2, 2}, dtype(DataType::kInt64).device(deviceType()));
    EXPECT_EQ(t_int64.element_size(), 8U);  // int64 = 8字节
}

// ==================== nbytes() 测试 ====================

TEST_P(TensorAttributesTest, NbytesBasic)
{
    // 测试基本字节数获取
    auto t = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t.nbytes(), 16U);  // 4个元素 × 4字节 = 16字节
    EXPECT_EQ(t.nbytes(), t.elements() * t.element_size());
}

TEST_P(TensorAttributesTest, NbytesDifferentShapes)
{
    // 测试不同形状的字节数
    // 1维，5个元素，float32
    auto t1 = Tensor::ones(Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t1.nbytes(), 20U);  // 5 × 4 = 20

    // 2维，2×3=6个元素，float32
    auto t2 = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t2.nbytes(), 24U);  // 6 × 4 = 24

    // 3维，2×3×4=24个元素，float32
    auto t3 = Tensor::ones(Shape{2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t3.nbytes(), 96U);  // 24 × 4 = 96
}

TEST_P(TensorAttributesTest, NbytesDifferentTypes)
{
    // 测试不同数据类型的字节数
    Shape shape{2, 2};  // 4个元素

    auto t_float32 = Tensor::ones(shape, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t_float32.nbytes(), 16U);  // 4 × 4 = 16

    auto t_float64 = Tensor::ones(shape, dtype(DataType::kFloat64).device(deviceType()));
    EXPECT_EQ(t_float64.nbytes(), 32U);  // 4 × 8 = 32

    auto t_int32 = Tensor::ones(shape, dtype(DataType::kInt32).device(deviceType()));
    EXPECT_EQ(t_int32.nbytes(), 16U);  // 4 × 4 = 16

    auto t_int64 = Tensor::ones(shape, dtype(DataType::kInt64).device(deviceType()));
    EXPECT_EQ(t_int64.nbytes(), 32U);  // 4 × 8 = 32
}

// ==================== 综合测试 ====================

TEST_P(TensorAttributesTest, AllAttributesTogether)
{
    // 测试所有属性一起使用
    Shape shape{3, 4};
    auto t = Tensor::ones(shape, dtype(DataType::kFloat32).device(deviceType()));

    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.ndim(), 2U);
    EXPECT_EQ(t.elements(), 12U);
    EXPECT_EQ(t.numel(), 12U);
    EXPECT_EQ(t.dtype(), DataType::kFloat32);
    EXPECT_EQ(t.device().type(), deviceType());
    EXPECT_EQ(t.element_size(), 4U);
    EXPECT_EQ(t.nbytes(), 48U);  // 12 × 4 = 48
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(TensorAttributesTest);
