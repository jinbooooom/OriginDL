#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#endif

using namespace origin;

/**
 * @brief 张量操作方法测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 *          无GPU环境只运行CPU测试，有GPU环境运行CPU+CUDA测试
 */
class TensorOperationsTest : public origin::test::OperatorTestBase
{};

// ==================== to() 类型转换测试 ====================

TEST_P(TensorOperationsTest, ToDataTypeConversion)
{
    // 测试数据类型转换
    auto t_float32 = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 转换为float64
    auto t_float64 = t_float32.to(DataType::kFloat64);
    EXPECT_EQ(t_float64.dtype(), DataType::kFloat64);
    EXPECT_EQ(t_float64.shape(), Shape({2, 2}));
    EXPECT_EQ(t_float64.device().type(), deviceType());

    // 转换为int32
    auto t_int32 = t_float32.to(DataType::kInt32);
    EXPECT_EQ(t_int32.dtype(), DataType::kInt32);
    EXPECT_EQ(t_int32.shape(), Shape({2, 2}));

    // 原张量应该保持不变
    EXPECT_EQ(t_float32.dtype(), DataType::kFloat32);
}

TEST_P(TensorOperationsTest, ToDeviceConversion)
{
    // 测试设备转换（双向：CPU<->CUDA）
    // 只有在CUDA可用时才执行双向转换测试
    if (!origin::test::TestUtils::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA is not available";
    }

    if (deviceType() == DeviceType::kCUDA)
    {
        // CUDA设备上：测试从CUDA转到CPU
        auto t_cuda = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

        auto t_cpu = t_cuda.to(Device(DeviceType::kCPU));
        EXPECT_EQ(t_cpu.device().type(), DeviceType::kCPU);
        EXPECT_EQ(t_cpu.shape(), Shape({2, 2}));
        EXPECT_EQ(t_cpu.dtype(), DataType::kFloat32);

        // 原张量应该仍在CUDA
        EXPECT_EQ(t_cuda.device().type(), DeviceType::kCUDA);

        // 再测试从CPU转回CUDA
        auto t_cuda2 = t_cpu.to(Device(DeviceType::kCUDA));
        EXPECT_EQ(t_cuda2.device().type(), DeviceType::kCUDA);
        EXPECT_EQ(t_cuda2.shape(), Shape({2, 2}));
        EXPECT_EQ(t_cuda2.dtype(), DataType::kFloat32);
    }
    else
    {
        // CPU设备上：测试从CPU转到CUDA
        auto t_cpu = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(DeviceType::kCPU));

        auto t_cuda = t_cpu.to(Device(DeviceType::kCUDA));
        EXPECT_EQ(t_cuda.device().type(), DeviceType::kCUDA);
        EXPECT_EQ(t_cuda.shape(), Shape({2, 2}));
        EXPECT_EQ(t_cuda.dtype(), DataType::kFloat32);

        // 原张量应该仍在CPU
        EXPECT_EQ(t_cpu.device().type(), DeviceType::kCPU);

        // 再测试从CUDA转回CPU
        auto t_cpu2 = t_cuda.to(Device(DeviceType::kCPU));
        EXPECT_EQ(t_cpu2.device().type(), DeviceType::kCPU);
        EXPECT_EQ(t_cpu2.shape(), Shape({2, 2}));
        EXPECT_EQ(t_cpu2.dtype(), DataType::kFloat32);
    }
}

TEST_P(TensorOperationsTest, ToTensorOptionsConversion)
{
    // 测试同时转换类型和设备
    // 注意：设备转换的双向测试已在 ToDeviceConversion 中覆盖，这里只测试类型+设备同时转换
    if (!origin::test::TestUtils::isCudaAvailable())
    {
        // 没有CUDA时，只测试类型转换
        auto t           = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto options     = TensorOptions().dtype(DataType::kFloat64).device(DeviceType::kCPU);
        auto t_converted = t.to(options);

        EXPECT_EQ(t_converted.dtype(), DataType::kFloat64);
        EXPECT_EQ(t_converted.device().type(), DeviceType::kCPU);
        EXPECT_EQ(t_converted.shape(), Shape({2, 2}));

        // 比较转换前后的值是否一致（转换为double类型比较，避免精度问题）
        auto original_values  = t.to(DataType::kFloat64).to_vector<double>();
        auto converted_values = t_converted.to_vector<double>();
        EXPECT_EQ(original_values.size(), converted_values.size());
        for (size_t i = 0; i < original_values.size(); ++i)
        {
            EXPECT_NEAR(original_values[i], converted_values[i], origin::test::TestTolerance::kDefault);
        }
        return;
    }

    if (deviceType() == DeviceType::kCUDA)
    {
        // CUDA设备上：测试从CUDA转到CPU（同时转换类型）
        auto t_cuda = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

        auto options_cpu = TensorOptions().dtype(DataType::kFloat64).device(DeviceType::kCPU);
        auto t_cpu       = t_cuda.to(options_cpu);

        EXPECT_EQ(t_cpu.dtype(), DataType::kFloat64);
        EXPECT_EQ(t_cpu.device().type(), DeviceType::kCPU);
        EXPECT_EQ(t_cpu.shape(), Shape({2, 2}));

        // 原张量应该保持不变
        EXPECT_EQ(t_cuda.dtype(), DataType::kFloat32);
        EXPECT_EQ(t_cuda.device().type(), DeviceType::kCUDA);

        // 比较转换前后的值是否一致（转换为double类型比较）
        auto original_values  = t_cuda.to(DataType::kFloat64).to_vector<double>();
        auto converted_values = t_cpu.to_vector<double>();
        EXPECT_EQ(original_values.size(), converted_values.size());
        for (size_t i = 0; i < original_values.size(); ++i)
        {
            EXPECT_NEAR(original_values[i], converted_values[i], origin::test::TestTolerance::kDefault);
        }
    }
    else
    {
        // CPU设备上：测试从CPU转到CUDA（同时转换类型）
        auto t_cpu = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(DeviceType::kCPU));

        auto options_cuda = TensorOptions().dtype(DataType::kFloat64).device(DeviceType::kCUDA);
        auto t_cuda       = t_cpu.to(options_cuda);

        EXPECT_EQ(t_cuda.dtype(), DataType::kFloat64);
        EXPECT_EQ(t_cuda.device().type(), DeviceType::kCUDA);
        EXPECT_EQ(t_cuda.shape(), Shape({2, 2}));

        // 原张量应该保持不变
        EXPECT_EQ(t_cpu.dtype(), DataType::kFloat32);
        EXPECT_EQ(t_cpu.device().type(), DeviceType::kCPU);

        // 比较转换前后的值是否一致（转换为double类型比较）
        auto original_values  = t_cpu.to(DataType::kFloat64).to_vector<double>();
        auto converted_values = t_cuda.to_vector<double>();
        EXPECT_EQ(original_values.size(), converted_values.size());
        for (size_t i = 0; i < original_values.size(); ++i)
        {
            EXPECT_NEAR(original_values[i], converted_values[i], origin::test::TestTolerance::kDefault);
        }
    }
}

TEST_P(TensorOperationsTest, ToSameTypeAndDevice)
{
    // 测试转换到相同类型和设备（应该仍然创建新张量）
    auto t      = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto t_same = t.to(dtype(DataType::kFloat32).device(deviceType()));

    EXPECT_EQ(t_same.dtype(), DataType::kFloat32);
    EXPECT_EQ(t_same.device().type(), deviceType());
    EXPECT_EQ(t_same.shape(), Shape({2, 2}));

    // 值应该相同
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(t, t_same, origin::test::TestTolerance::kDefault);
}

// ==================== item() 测试 ====================

TEST_P(TensorOperationsTest, ItemZeroDimensionalTensor)
{
    // 测试0维张量（标量张量，形状为 {}）
    auto t = Tensor::full(Shape{}, 3.14f, dtype(DataType::kFloat32).device(deviceType()));

    // 验证是0维张量
    EXPECT_EQ(t.ndim(), 0U);
    EXPECT_EQ(t.shape().size(), 0U);
    EXPECT_TRUE(t.shape().is_scalar());
    EXPECT_EQ(t.elements(), 1U);

    // 测试item()取值
    float value = t.item<float>();
    EXPECT_NEAR(value, 3.14f, origin::test::TestTolerance::kDefault);
}

TEST_P(TensorOperationsTest, ItemSingleElementTensor)
{
    // 测试单元素张量（1维，形状为 {1}）
    auto t = Tensor({42.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    // 验证是1维单元素张量
    EXPECT_EQ(t.ndim(), 1U);
    EXPECT_EQ(t.shape().size(), 1U);
    EXPECT_EQ(t.shape()[0], 1U);
    EXPECT_EQ(t.elements(), 1U);

    // 测试item()取值
    float value = t.item<float>();
    EXPECT_NEAR(value, 42.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(TensorOperationsTest, ItemDifferentTypes)
{
    // 测试不同数据类型的item（使用0维张量）
    auto t_float32 = Tensor::full(Shape{}, 3.14f, dtype(DataType::kFloat32).device(deviceType()));
    float f32_val  = t_float32.item<float>();
    EXPECT_NEAR(f32_val, 3.14f, origin::test::TestTolerance::kDefault);

    auto t_float64 = Tensor::full(Shape{}, 3.14159, dtype(DataType::kFloat64).device(deviceType()));
    double f64_val = t_float64.item<double>();
    EXPECT_NEAR(f64_val, 3.14159, origin::test::TestTolerance::kDefault);

    auto t_int32    = Tensor::full(Shape{}, 42, dtype(DataType::kInt32).device(deviceType()));
    int32_t i32_val = t_int32.item<int32_t>();
    EXPECT_EQ(i32_val, 42);
}

TEST_P(TensorOperationsTest, ItemThrowsOnNonScalarTensor)
{
    // 测试非法情况：item() 只能用于标量张量（元素数量为1）
    // 如果元素数量大于1，应该抛出异常

    // 测试2维多元素张量（2x2，4个元素）
    auto t_2d = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_THROW(t_2d.item<float>(), std::exception);

    // 测试1维多元素张量（10个元素）
    auto t_1d = Tensor::ones(Shape{10}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_THROW(t_1d.item<float>(), std::exception);

    // 测试3维张量（2x2x2，8个元素）
    auto t_3d = Tensor::ones(Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_THROW(t_3d.item<float>(), std::exception);
}

// ==================== data_ptr() 测试 ====================

TEST_P(TensorOperationsTest, DataPtrBasic)
{
    // 测试基本指针访问
    auto t = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 注意：CUDA张量的data_ptr在CPU代码中直接访问可能导致问题
    // 这里只测试CPU版本，或者先将CUDA张量复制到CPU
    if (deviceType() == DeviceType::kCPU)
    {
        float *ptr = t.data_ptr<float>();
        EXPECT_NE(ptr, nullptr);
        EXPECT_NEAR(ptr[0], 1.0f, origin::test::TestTolerance::kDefault);
        EXPECT_NEAR(ptr[1], 2.0f, origin::test::TestTolerance::kDefault);
        EXPECT_NEAR(ptr[2], 3.0f, origin::test::TestTolerance::kDefault);
        EXPECT_NEAR(ptr[3], 4.0f, origin::test::TestTolerance::kDefault);
    }
    else
    {
#ifdef WITH_CUDA
        // CUDA版本：直接获取CUDA张量的指针，使用cudaMemcpy拷贝到CPU内存进行比较
        float *cuda_ptr = t.data_ptr<float>();
        EXPECT_NE(cuda_ptr, nullptr);

        // 获取元素数量
        size_t num_elements = t.elements();
        size_t data_size    = num_elements * sizeof(float);

        // 分配CPU内存用于接收数据
        std::vector<float> cpu_data(num_elements);

        // 使用cudaMemcpy从GPU内存拷贝到CPU内存
        cudaError_t err = cudaMemcpy(cpu_data.data(), cuda_ptr, data_size, cudaMemcpyDeviceToHost);
        EXPECT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);

        // 同步等待拷贝完成
        cuda::synchronize();

        // 比较数据
        std::vector<float> expected_data = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_EQ(cpu_data.size(), expected_data.size());
        for (size_t i = 0; i < cpu_data.size(); ++i)
        {
            EXPECT_NEAR(cpu_data[i], expected_data[i], origin::test::TestTolerance::kDefault);
        }
#else
        // TORCH后端或其他后端：跳过CUDA特定测试
        GTEST_SKIP() << "CUDA support not enabled, skipping CUDA-specific test";
#endif
    }
}

TEST_P(TensorOperationsTest, DataPtrModification)
{
    // 测试通过指针修改数据
    auto t = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    if (deviceType() == DeviceType::kCPU)
    {
        float *ptr = t.data_ptr<float>();
        ptr[0]     = 10.0f;
        ptr[1]     = 20.0f;

        // 验证修改生效
        auto data = t.to_vector<float>();
        EXPECT_NEAR(data[0], 10.0f, origin::test::TestTolerance::kDefault);
        EXPECT_NEAR(data[1], 20.0f, origin::test::TestTolerance::kDefault);
    }
    else
    {
#ifdef WITH_CUDA
        // CUDA版本：使用cudaMemcpy在CPU和GPU之间传输数据
        float *cuda_ptr = t.data_ptr<float>();
        EXPECT_NE(cuda_ptr, nullptr);

        // 获取元素数量
        size_t num_elements = t.elements();
        size_t data_size    = num_elements * sizeof(float);

        // 分配CPU内存用于修改数据
        std::vector<float> cpu_data(num_elements);

        // 从GPU拷贝到CPU
        cudaError_t err = cudaMemcpy(cpu_data.data(), cuda_ptr, data_size, cudaMemcpyDeviceToHost);
        EXPECT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
        cuda::synchronize();

        // 在CPU上修改数据
        cpu_data[0] = 10.0f;
        cpu_data[1] = 20.0f;

        // 从CPU拷贝回GPU
        err = cudaMemcpy(cuda_ptr, cpu_data.data(), data_size, cudaMemcpyHostToDevice);
        EXPECT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
        cuda::synchronize();

        // 验证修改生效：使用to_vector获取数据进行比较
        auto verify_data = t.to_vector<float>();
        EXPECT_EQ(verify_data.size(), num_elements);
        EXPECT_NEAR(verify_data[0], 10.0f, origin::test::TestTolerance::kDefault);
        EXPECT_NEAR(verify_data[1], 20.0f, origin::test::TestTolerance::kDefault);
#else
        // TORCH后端或其他后端：跳过CUDA特定测试
        GTEST_SKIP() << "CUDA support not enabled, skipping CUDA-specific test";
#endif
    }
}

// ==================== to_vector() 测试 ====================

TEST_P(TensorOperationsTest, ToVectorBasic)
{
    // 测试基本向量转换
    std::vector<float> original_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor(original_data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto vec = t.to_vector<float>();
    EXPECT_EQ(vec.size(), 4U);
    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(vec[i], original_data[i], origin::test::TestTolerance::kDefault);
    }
}

TEST_P(TensorOperationsTest, ToVectorDifferentShapes)
{
    // 测试不同形状的向量转换
    // 1维
    std::vector<float> expected1 = {1.0f, 2.0f, 3.0f};
    auto t1                      = Tensor(expected1, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto vec1                    = t1.to_vector<float>();
    EXPECT_EQ(vec1.size(), 3U);
    EXPECT_EQ(vec1.size(), expected1.size());
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        EXPECT_NEAR(vec1[i], expected1[i], origin::test::TestTolerance::kDefault);
    }

    // 2维
    std::vector<float> expected2 = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t2                      = Tensor(expected2, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto vec2                    = t2.to_vector<float>();
    EXPECT_EQ(vec2.size(), 4U);
    EXPECT_EQ(vec2.size(), expected2.size());
    for (size_t i = 0; i < vec2.size(); ++i)
    {
        EXPECT_NEAR(vec2[i], expected2[i], origin::test::TestTolerance::kDefault);
    }

    // 3维
    auto t3   = Tensor::ones(Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto vec3 = t3.to_vector<float>();
    EXPECT_EQ(vec3.size(), 8U);
    // 验证所有元素都是1.0
    for (size_t i = 0; i < vec3.size(); ++i)
    {
        EXPECT_NEAR(vec3[i], 1.0f, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(TensorOperationsTest, ToVectorDifferentTypes)
{
    // 测试不同数据类型的向量转换
    auto t_float32 = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto vec_f32   = t_float32.to_vector<float>();
    EXPECT_EQ(vec_f32.size(), 4U);
    for (size_t i = 0; i < vec_f32.size(); ++i)
    {
        EXPECT_NEAR(vec_f32[i], 1.0f, origin::test::TestTolerance::kDefault);
    }

    auto t_int32 = Tensor::ones(Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    auto vec_i32 = t_int32.to_vector<int32_t>();
    EXPECT_EQ(vec_i32.size(), 4U);
    for (size_t i = 0; i < vec_i32.size(); ++i)
    {
        EXPECT_EQ(vec_i32[i], 1);
    }

    // 测试类型转换：将float32类型的张量转换为int32类型的vector（先转换dtype，再dump）
    auto t_float = Tensor({1.5f, 2.7f, 3.2f, 4.9f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto t_int   = t_float.to(DataType::kInt32);
    auto vec_int = t_int.to_vector<int32_t>();
    EXPECT_EQ(vec_int.size(), 4U);
    // 验证转换后的值（浮点数转整数会截断）
    EXPECT_EQ(vec_int[0], 1);  // 1.5 -> 1
    EXPECT_EQ(vec_int[1], 2);  // 2.7 -> 2
    EXPECT_EQ(vec_int[2], 3);  // 3.2 -> 3
    EXPECT_EQ(vec_int[3], 4);  // 4.9 -> 4

    // 测试将int32类型的张量转换为float32类型的vector（先转换dtype，再dump）
    auto t_int2    = Tensor({1, 2, 3, 4}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    auto t_float2  = t_int2.to(DataType::kFloat32);
    auto vec_float = t_float2.to_vector<float>();
    EXPECT_EQ(vec_float.size(), 4U);
    for (size_t i = 0; i < vec_float.size(); ++i)
    {
        EXPECT_NEAR(vec_float[i], static_cast<float>(i + 1), origin::test::TestTolerance::kDefault);
    }
}

TEST_P(TensorOperationsTest, ToVectorLargeTensor)
{
    // 测试大张量的向量转换
    std::vector<float> data(100, 1.0f);
    auto t = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto vec = t.to_vector<float>();
    EXPECT_EQ(vec.size(), 100U);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        EXPECT_NEAR(vec[i], 1.0f, origin::test::TestTolerance::kDefault);
    }
}

// ==================== 综合测试 ====================

TEST_P(TensorOperationsTest, OperationsChain)
{
    // 测试操作链：组合多种张量操作
    // 1. 创建张量
    auto t = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    EXPECT_EQ(t.shape(), Shape({2, 2}));
    EXPECT_EQ(t.dtype(), DataType::kFloat32);
    EXPECT_EQ(t.device().type(), deviceType());

    // 2. 转换类型
    auto t_float64 = t.to(DataType::kFloat64);
    EXPECT_EQ(t_float64.dtype(), DataType::kFloat64);
    EXPECT_EQ(t_float64.shape(), Shape({2, 2}));
    EXPECT_EQ(t_float64.device().type(), deviceType());

    // 3. 转换为向量并验证数据
    auto vec = t_float64.to_vector<double>();
    EXPECT_EQ(vec.size(), 4U);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        EXPECT_NEAR(vec[i], 1.0, origin::test::TestTolerance::kDefault);
    }

    // 4. 创建0维标量张量并获取值
    auto scalar = Tensor::full(Shape{}, 42.0, dtype(DataType::kFloat64).device(deviceType()));
    EXPECT_EQ(scalar.ndim(), 0U);
    EXPECT_EQ(scalar.elements(), 1U);
    double value = scalar.item<double>();
    EXPECT_NEAR(value, 42.0, origin::test::TestTolerance::kDefault);

    // 5. 测试设备转换（如果有CUDA可用）
    if (origin::test::TestUtils::isCudaAvailable())
    {
        auto t_cpu  = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto t_cuda = t_cpu.to(Device(DeviceType::kCUDA));
        EXPECT_EQ(t_cuda.device().type(), DeviceType::kCUDA);

        // 转换回CPU并验证数据
        auto t_back   = t_cuda.to(Device(DeviceType::kCPU));
        auto vec_back = t_back.to_vector<float>();
        EXPECT_EQ(vec_back.size(), 3U);
        for (size_t i = 0; i < vec_back.size(); ++i)
        {
            EXPECT_NEAR(vec_back[i], 1.0f, origin::test::TestTolerance::kDefault);
        }
    }

    // 6. 测试类型转换和向量转换的组合
    auto t_mixed = Tensor({1.5f, 2.5f, 3.5f, 4.5f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto t_int   = t_mixed.to(DataType::kInt32);
    auto vec_int = t_int.to_vector<int32_t>();
    EXPECT_EQ(vec_int.size(), 4U);
    EXPECT_EQ(vec_int[0], 1);  // 1.5 -> 1
    EXPECT_EQ(vec_int[1], 2);  // 2.5 -> 2
    EXPECT_EQ(vec_int[2], 3);  // 3.5 -> 3
    EXPECT_EQ(vec_int[3], 4);  // 4.5 -> 4
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(TensorOperationsTest);
