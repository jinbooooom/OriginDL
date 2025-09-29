#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "dlTensor.h"

using namespace dl;

class TensorCreateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试前的设置
    }
    
    void TearDown() override {
        // 测试后的清理
    }
    
    // 辅助函数：比较两个浮点数是否相等（考虑浮点精度）
    bool isClose(double a, double b, double tolerance = 1e-6) {
        return std::abs(a - b) < tolerance;
    }
    
    // 辅助函数：比较两个Tensor是否相等
    bool tensorsEqual(const Tensor& a, const Tensor& b, double tolerance = 1e-6) {
        if (a.shape() != b.shape()) {
            return false;
        }
        
        auto data_a = a.to_vector();
        auto data_b = b.to_vector();
        
        if (data_a.size() != data_b.size()) {
            return false;
        }
        
        for (size_t i = 0; i < data_a.size(); ++i) {
            if (!isClose(data_a[i], data_b[i], tolerance)) {
                return false;
            }
        }
        return true;
    }
};

// ==================== Tensor构造测试 ====================

TEST_F(TensorCreateTest, ConstructorFromVector) {
    // 测试从vector构造Tensor
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};
    Tensor tensor(data, shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.ndim(), 2U);
    EXPECT_EQ(tensor.elements(), 4U);
    
    auto result_data = tensor.to_vector();
    EXPECT_EQ(result_data.size(), 4U);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(result_data[i], data[i]);
    }
}

TEST_F(TensorCreateTest, ConstructorFromInitializerList) {
    // 测试从初始化列表构造Tensor
    Shape shape{2, 2};
    Tensor tensor({1.0, 2.0, 3.0, 4.0}, shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 4U);
    
    auto result_data = tensor.to_vector();
    std::vector<data_t> expected = {1.0, 2.0, 3.0, 4.0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(result_data[i], expected[i]);
    }
}

TEST_F(TensorCreateTest, ConstructorFromScalar) {
    // 测试从标量构造Tensor
    Shape shape{3, 3};
    data_t scalar = 5.0;
    Tensor tensor(scalar, shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 9U);
    
    auto result_data = tensor.to_vector();
    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_DOUBLE_EQ(result_data[i], scalar);
    }
}

TEST_F(TensorCreateTest, CopyConstructor) {
    // 测试拷贝构造函数
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};
    Tensor original(data, shape);
    Tensor copy(original);
    
    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_TRUE(tensorsEqual(copy, original));
}

TEST_F(TensorCreateTest, MoveConstructor) {
    // 测试移动构造函数
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};
    Tensor original(data, shape);
    Tensor moved(std::move(original));
    
    EXPECT_EQ(moved.shape(), shape);
    auto result_data = moved.to_vector();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(result_data[i], data[i]);
    }
}

TEST_F(TensorCreateTest, AssignmentOperators) {
    // 测试赋值运算符
    std::vector<data_t> data1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_t> data2 = {5.0, 6.0, 7.0, 8.0};
    Shape shape{2, 2};
    
    Tensor tensor1(data1, shape);
    Tensor tensor2(data2, shape);
    
    // 拷贝赋值
    tensor1 = tensor2;
    EXPECT_TRUE(tensorsEqual(tensor1, tensor2));
    
    // 移动赋值
    Tensor tensor3(data1, shape);
    tensor3 = std::move(tensor2);
    EXPECT_EQ(tensor3.shape(), shape);
}

TEST_F(TensorCreateTest, FactoryMethods) {
    // 测试工厂方法
    Shape shape{2, 3};
    
    // 测试zeros
    Tensor zeros_tensor = Tensor::zeros(shape);
    EXPECT_EQ(zeros_tensor.shape(), shape);
    auto zeros_data = zeros_tensor.to_vector();
    for (auto val : zeros_data) {
        EXPECT_DOUBLE_EQ(val, 0.0);
    }
    
    // 测试ones
    Tensor ones_tensor = Tensor::ones(shape);
    EXPECT_EQ(ones_tensor.shape(), shape);
    auto ones_data = ones_tensor.to_vector();
    for (auto val : ones_data) {
        EXPECT_DOUBLE_EQ(val, 1.0);
    }
    
    // 测试constant
    data_t constant_val = 3.14;
    Tensor constant_tensor = Tensor::constant(constant_val, shape);
    EXPECT_EQ(constant_tensor.shape(), shape);
    auto constant_data = constant_tensor.to_vector();
    for (auto val : constant_data) {
        EXPECT_DOUBLE_EQ(val, constant_val);
    }
    
}

TEST_F(TensorCreateTest, RandnFactory) {
    // 测试随机张量生成
    Shape shape{3, 3};
    Tensor rand_tensor = Tensor::randn(shape);
    
    EXPECT_EQ(rand_tensor.shape(), shape);
    EXPECT_EQ(rand_tensor.elements(), 9U);
    
    // 验证生成的随机数在合理范围内（正态分布）
    auto rand_data = rand_tensor.to_vector();
    for (auto val : rand_data) {
        // 正态分布的值通常在[-3, 3]范围内
        EXPECT_TRUE(std::abs(val) < 10.0);
    }
}

TEST_F(TensorCreateTest, ShapeValidation) {
    // 测试形状验证
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape valid_shape{2, 2};
    Shape invalid_shape{3, 3};  // 数据大小不匹配
    
    // 有效形状应该成功
    EXPECT_NO_THROW(Tensor tensor(data, valid_shape));
    
    // 无效形状应该抛出异常
    EXPECT_THROW(Tensor tensor(data, invalid_shape), std::exception);
}

TEST_F(TensorCreateTest, EmptyTensor) {
    // 测试空张量
    std::vector<data_t> data;
    Shape shape{0, 0};
    Tensor tensor(data, shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 0U);
    EXPECT_EQ(tensor.ndim(), 2U);
}

TEST_F(TensorCreateTest, ScalarTensor) {
    // 测试标量张量
    std::vector<data_t> data = {3.14};
    Shape shape{1, 1};
    Tensor tensor(data, shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 1U);
    EXPECT_DOUBLE_EQ(tensor.item(), 3.14);
}

TEST_F(TensorCreateTest, LargeTensor) {
    // 测试大张量
    std::vector<data_t> data(1000, 1.0);
    Shape shape{100, 10};
    Tensor tensor(data, shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.elements(), 1000U);
    
    auto result_data = tensor.to_vector();
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_DOUBLE_EQ(result_data[i], 1.0);
    }
}

TEST_F(TensorCreateTest, OneDimensionalTensor) {
    // 测试一维张量
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    Shape shape{5, 1};
    Tensor tensor(data, shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.ndim(), 2U);
    EXPECT_EQ(tensor.elements(), 5U);
    
    auto result_data = tensor.to_vector();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(result_data[i], data[i]);
    }
}

TEST_F(TensorCreateTest, ThreeDimensionalTensor) {
    // 测试三维张量
    std::vector<data_t> data(24, 1.0);  // 2*3*4 = 24
    Shape shape{2, 3, 4};
    Tensor tensor(data, shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.ndim(), 3U);
    EXPECT_EQ(tensor.elements(), 24U);
    
    auto result_data = tensor.to_vector();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(result_data[i], data[i]);
    }
}

TEST_F(TensorCreateTest, DataIntegrity) {
    // 测试数据完整性
    std::vector<data_t> original_data = {1.1, 2.2, 3.3, 4.4, 5.5};
    Shape shape{5, 1};
    Tensor tensor(original_data, shape);
    
    auto retrieved_data = tensor.to_vector();
    
    EXPECT_EQ(original_data.size(), retrieved_data.size());
    for (size_t i = 0; i < original_data.size(); ++i) {
        EXPECT_DOUBLE_EQ(original_data[i], retrieved_data[i]);
    }
}

TEST_F(TensorCreateTest, MemoryManagement) {
    // 测试内存管理
    std::vector<data_t> data = {1.0, 2.0, 3.0, 4.0};
    Shape shape{2, 2};
    
    {
        Tensor tensor1(data, shape);
        Tensor tensor2 = tensor1;  // 拷贝构造
        Tensor tensor3 = std::move(tensor1);  // 移动构造
        
        EXPECT_EQ(tensor2.shape(), shape);
        EXPECT_EQ(tensor3.shape(), shape);
        
        // 验证数据正确性
        auto data2 = tensor2.to_vector();
        auto data3 = tensor3.to_vector();
        
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_DOUBLE_EQ(data2[i], data[i]);
            EXPECT_DOUBLE_EQ(data3[i], data[i]);
        }
    }
    // 离开作用域后，张量应该被正确销毁
}
