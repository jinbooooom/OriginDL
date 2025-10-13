#include <gtest/gtest.h>
#include <vector>
#include "origin/core/tensor.h"
#include "origin/mat/basic_types.h"

using namespace origin;

class MultiTypeTensorTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// 测试自动类型推断
TEST_F(MultiTypeTensorTest, AutoTypeInference)
{
    // 测试float32类型推断（使用std::initializer_list）
    Tensor t1({1.0f, 2.0f, 3.0f}, Shape{3});
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);

    // 测试int32类型推断（使用std::initializer_list）
    Tensor t2({1, 2, 3}, Shape{3});
    EXPECT_EQ(t2.dtype(), DataType::kInt32);

    // 测试int8类型推断（使用std::initializer_list）
    Tensor t3({1, 2, 3}, Shape{3});
    EXPECT_EQ(t3.dtype(), DataType::kInt32);  // 注意：int字面量默认是int32

    // 测试vector构造函数
    std::vector<float> float_vec = {1.0f, 2.0f, 3.0f};
    Tensor t4(float_vec, Shape{3});
    EXPECT_EQ(t4.dtype(), DataType::kFloat32);

    std::vector<int32_t> int_vec = {1, 2, 3};
    Tensor t5(int_vec, Shape{3});
    EXPECT_EQ(t5.dtype(), DataType::kInt32);
}

// 测试类型转换
TEST_F(MultiTypeTensorTest, TypeConversion)
{
    // 创建float32张量
    Tensor t1({1.5f, 2.7f, 3.2f}, Shape{3});
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);

    // 转换为int32
    Tensor t2 = t1.to(DataType::kInt32);
    EXPECT_EQ(t2.dtype(), DataType::kInt32);

    // 转换为int8
    Tensor t3 = t1.to(DataType::kInt8);
    EXPECT_EQ(t3.dtype(), DataType::kInt8);

    // 测试反向转换
    Tensor t4({1, 2, 3}, Shape{3});
    EXPECT_EQ(t4.dtype(), DataType::kInt32);

    Tensor t5 = t4.to(DataType::kFloat32);
    EXPECT_EQ(t5.dtype(), DataType::kFloat32);
}

// 测试标量张量
TEST_F(MultiTypeTensorTest, ScalarTensors)
{
    // 测试float标量
    Tensor t1(5.0f, Shape{1});
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.shape(), (Shape{1}));

    // 测试int标量
    Tensor t2(42, Shape{1});
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.shape(), (Shape{1}));

    // 测试int8标量
    Tensor t3(static_cast<int8_t>(10), Shape{1});
    EXPECT_EQ(t3.dtype(), DataType::kInt8);
    EXPECT_EQ(t3.shape(), (Shape{1}));
}

// 测试指定数据类型的构造函数
TEST_F(MultiTypeTensorTest, ExplicitTypeConstructor)
{
    std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
    std::vector<int32_t> int_data = {4, 5, 6};

    // 使用from_blob方法指定类型
    Tensor t1 = Tensor::from_blob(float_data.data(), Shape{3}, dtype(DataType::kFloat32));
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);

    Tensor t2 = Tensor::from_blob(int_data.data(), Shape{3}, dtype(DataType::kInt32));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);

    // 使用标量构造函数指定类型
    Tensor t3(5.0, Shape{1}, DataType::kFloat32);
    EXPECT_EQ(t3.dtype(), DataType::kFloat32);

    Tensor t4(42.0, Shape{1}, DataType::kInt32);
    EXPECT_EQ(t4.dtype(), DataType::kInt32);

    // 测试默认dtype参数
    Tensor t5(float_data, Shape{3});  // 默认应该是kFloat32
    EXPECT_EQ(t5.dtype(), DataType::kFloat32);

    Tensor t6(5.0f, Shape{1});  // 默认应该是kFloat32
    EXPECT_EQ(t6.dtype(), DataType::kFloat32);

    Tensor t7(5.0, Shape{1}); // 没有加 .f，当做 double 处理
    EXPECT_EQ(t7.dtype(), DataType::kDouble);

    Tensor t8(5.0, Shape{1}, DataType::kInt32); // 指定类型，当做 int32 处理
    EXPECT_EQ(t8.dtype(), DataType::kInt32);

    Tensor t9(5.0, Shape{1}, DataType::kInt8);
    EXPECT_EQ(t9.dtype(), DataType::kInt8);

    Tensor t10(float_data, Shape{3}, DataType::kFloat32);
    EXPECT_EQ(t10.dtype(), DataType::kFloat32);

    Tensor t11(int_data, Shape{3}, DataType::kInt32);
    EXPECT_EQ(t11.dtype(), DataType::kInt32);

    Tensor t12(int_data, Shape{3}, DataType::kInt8);
    EXPECT_EQ(t12.dtype(), DataType::kInt8);
}

// 测试工厂函数
TEST_F(MultiTypeTensorTest, FactoryFunctions)
{
    // 测试zeros工厂函数
    Tensor t1 = Tensor::zeros(Shape{2, 3}, dtype(DataType::kFloat32));  // 默认kFloat32
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.shape(), (Shape{2, 3}));

    Tensor t2 = Tensor::zeros(Shape{2, 3}, dtype(DataType::kInt32));
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.shape(), (Shape{2, 3}));

    Tensor t3 = Tensor::zeros(Shape{2, 3}, dtype(DataType::kInt8));
    EXPECT_EQ(t3.dtype(), DataType::kInt8);
    EXPECT_EQ(t3.shape(), (Shape{2, 3}));

    // 测试ones工厂函数
    Tensor t4 = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32));  // 默认kFloat32
    EXPECT_EQ(t4.dtype(), DataType::kFloat32);
    EXPECT_EQ(t4.shape(), (Shape{2, 2}));

    Tensor t5 = Tensor::ones(Shape{2, 2}, dtype(DataType::kInt32));
    EXPECT_EQ(t5.dtype(), DataType::kInt32);
    EXPECT_EQ(t5.shape(), (Shape{2, 2}));

    // 测试randn工厂函数
    Tensor t6 = Tensor::randn(Shape{3, 3}, dtype(DataType::kFloat32));  // 默认kFloat32
    EXPECT_EQ(t6.dtype(), DataType::kFloat32);
    EXPECT_EQ(t6.shape(), (Shape{3, 3}));

    Tensor t7 = Tensor::randn(Shape{3, 3}, dtype(DataType::kInt32));
    EXPECT_EQ(t7.dtype(), DataType::kInt32);
    EXPECT_EQ(t7.shape(), (Shape{3, 3}));
}

// 测试std::initializer_list构造函数
TEST_F(MultiTypeTensorTest, InitializerListConstructor)
{
    // 测试float类型的initializer_list
    Tensor t1({1.0f, 2.0f, 3.0f}, Shape{3});
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t1.shape(), (Shape{3}));

    // 测试int类型的initializer_list
    Tensor t2({1, 2, 3, 4}, Shape{2, 2});
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t2.shape(), (Shape{2, 2}));

    // 测试int8类型的initializer_list
    Tensor t3({static_cast<int8_t>(1), static_cast<int8_t>(2)}, Shape{2});
    EXPECT_EQ(t3.dtype(), DataType::kInt8);
    EXPECT_EQ(t3.shape(), (Shape{2}));

    // 测试与vector构造函数的一致性
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    Tensor t4(vec, Shape{3});
    Tensor t5({1.0f, 2.0f, 3.0f}, Shape{3});
    EXPECT_EQ(t4.dtype(), t5.dtype());
    EXPECT_EQ(t4.shape(), t5.shape());
}

// 测试向后兼容性
TEST_F(MultiTypeTensorTest, BackwardCompatibility)
{
    // 测试原有的构造函数仍然工作（现在使用模板版本）
    std::vector<data_t> data = {1.0f, 2.0f, 3.0f};
    Tensor t1(data, Shape{3});
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);

    // 测试原有的标量构造函数（现在使用模板版本）
    Tensor t2(5.0f, Shape{1});
    EXPECT_EQ(t2.dtype(), DataType::kFloat32);

    // 测试原有的initializer_list构造函数（现在使用模板版本）
    Tensor t3({1.0f, 2.0f, 3.0f}, Shape{3});
    EXPECT_EQ(t3.dtype(), DataType::kFloat32);
}

// 测试类型查询
TEST_F(MultiTypeTensorTest, TypeQuery)
{
    Tensor t1({1.0f, 2.0f}, Shape{2});
    Tensor t2({1, 2}, Shape{2});
    Tensor t3({1, 2}, Shape{2});

    // 测试dtype()方法
    EXPECT_EQ(t1.dtype(), DataType::kFloat32);
    EXPECT_EQ(t2.dtype(), DataType::kInt32);
    EXPECT_EQ(t3.dtype(), DataType::kInt32);

    // 测试类型比较
    EXPECT_TRUE(t1.dtype() == DataType::kFloat32);
    EXPECT_TRUE(t2.dtype() == DataType::kInt32);
    EXPECT_FALSE(t1.dtype() == DataType::kInt32);
}

// 测试错误处理
TEST_F(MultiTypeTensorTest, ErrorHandling)
{
    // 测试不支持的类型转换（如果有的话）
    // 这里可以添加更多的错误处理测试
    EXPECT_NO_THROW({
        Tensor t1({1.0f, 2.0f}, Shape{2});
        Tensor t2 = t1.to(DataType::kInt32);
    });
}

// 测试形状和元素数量
TEST_F(MultiTypeTensorTest, ShapeAndElements)
{
    Tensor t1({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    EXPECT_EQ(t1.shape(), (Shape{2, 2}));
    EXPECT_EQ(t1.elements(), 4);
    EXPECT_EQ(t1.ndim(), 2);

    Tensor t2(5.0f, Shape{1});
    EXPECT_EQ(t2.shape(), (Shape{1}));
    EXPECT_EQ(t2.elements(), 1);
    EXPECT_EQ(t2.ndim(), 1);
}
