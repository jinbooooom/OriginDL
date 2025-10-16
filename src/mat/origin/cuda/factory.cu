#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"
#include <random>
#include <curand.h>

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA随机数生成器初始化
 */
void init_curand_generator(curandGenerator_t *gen)
{
    curandStatus_t status = curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS)
    {
        THROW_RUNTIME_ERROR("Failed to create CURAND generator: {}", static_cast<int>(status));
    }
    
    // 使用当前时间作为种子
    status = curandSetPseudoRandomGeneratorSeed(*gen, time(nullptr));
    if (status != CURAND_STATUS_SUCCESS)
    {
        curandDestroyGenerator(*gen);
        THROW_RUNTIME_ERROR("Failed to set CURAND seed: {}", static_cast<int>(status));
    }
}

/**
 * @brief 在CUDA设备上创建随机张量
 */
std::unique_ptr<OriginMat> randn(const Shape &shape, const TensorOptions &options)
{
    // 验证设备
    if (options.device().type() != DeviceType::kCUDA)
    {
        THROW_INVALID_ARG("CUDA randn requires CUDA device, got: {}", options.device().to_string());
    }
    
    // 设置CUDA设备
    set_cuda_device(options.device().index());
    
    auto result = std::unique_ptr<OriginMat>(new OriginMat(shape, options.dtype(), options.device()));
    
    // 获取数据指针
    void *data = result->storage()->data();
    size_t n = shape.elements();
    
    // 创建CURAND生成器
    curandGenerator_t gen;
    init_curand_generator(&gen);
    
    // 根据数据类型生成随机数
    curandStatus_t status;
    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            status = curandGenerateNormal(gen, static_cast<float*>(data), n, 0.0f, 1.0f);
            break;
        }
        case DataType::kFloat64:
        {
            status = curandGenerateNormalDouble(gen, static_cast<double*>(data), n, 0.0, 1.0);
            break;
        }
        case DataType::kInt32:
        {
            // 对于整数类型，先生成浮点数，然后转换
            std::vector<float> temp_data(n);
            status = curandGenerateNormal(gen, temp_data.data(), n, 0.0f, 1.0f);
            if (status == CURAND_STATUS_SUCCESS)
            {
                // 将浮点数转换为整数并复制到设备
                std::vector<int32_t> int_data(n);
                for (size_t i = 0; i < n; ++i)
                {
                    int_data[i] = static_cast<int32_t>(temp_data[i]);
                }
                cudaMemcpy(data, int_data.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
            }
            break;
        }
        case DataType::kInt8:
        {
            // 对于整数类型，先生成浮点数，然后转换
            std::vector<float> temp_data(n);
            status = curandGenerateNormal(gen, temp_data.data(), n, 0.0f, 1.0f);
            if (status == CURAND_STATUS_SUCCESS)
            {
                // 将浮点数转换为整数并复制到设备
                std::vector<int8_t> int_data(n);
                for (size_t i = 0; i < n; ++i)
                {
                    int_data[i] = static_cast<int8_t>(temp_data[i]);
                }
                cudaMemcpy(data, int_data.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice);
            }
            break;
        }
        default:
            curandDestroyGenerator(gen);
            THROW_INVALID_ARG("Unsupported data type {} for CUDA randn operation", dtype_to_string(options.dtype()));
    }
    
    // 清理生成器
    curandDestroyGenerator(gen);
    
    if (status != CURAND_STATUS_SUCCESS)
    {
        THROW_RUNTIME_ERROR("CURAND generation failed: {}", static_cast<int>(status));
    }
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    return result;
}

/**
 * @brief 在CUDA设备上创建零张量
 */
std::unique_ptr<OriginMat> zeros(const Shape &shape, const TensorOptions &options)
{
    // 验证设备
    if (options.device().type() != DeviceType::kCUDA)
    {
        THROW_INVALID_ARG("CUDA zeros requires CUDA device, got: {}", options.device().to_string());
    }
    
    // 设置CUDA设备
    set_cuda_device(options.device().index());
    
    auto result = std::unique_ptr<OriginMat>(new OriginMat(shape, options.dtype(), options.device()));
    
    // 获取数据指针
    void *data = result->storage()->data();
    size_t n = shape.elements();
    
    // 使用cudaMemset清零
    cudaError_t err = cudaMemset(data, 0, n * get_type_size(options.dtype()));
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA memset failed: {}", cudaGetErrorString(err));
    }
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    return result;
}

/**
 * @brief 在CUDA设备上创建全1张量
 */
std::unique_ptr<OriginMat> ones(const Shape &shape, const TensorOptions &options)
{
    // 验证设备
    if (options.device().type() != DeviceType::kCUDA)
    {
        THROW_INVALID_ARG("CUDA ones requires CUDA device, got: {}", options.device().to_string());
    }
    
    // 设置CUDA设备
    set_cuda_device(options.device().index());
    
    auto result = std::unique_ptr<OriginMat>(new OriginMat(shape, options.dtype(), options.device()));
    
    // 获取数据指针
    void *data = result->storage()->data();
    size_t n = shape.elements();
    
    // 根据数据类型设置值
    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            float value = 1.0f;
            cudaError_t err = cudaMemset(data, 0, n * sizeof(float));
            if (err == cudaSuccess)
            {
                // 使用CUDA kernel设置所有元素为1
                // 这里简化处理，实际应该使用kernel
                std::vector<float> ones_data(n, 1.0f);
                cudaMemcpy(data, ones_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            }
            break;
        }
        case DataType::kFloat64:
        {
            std::vector<double> ones_data(n, 1.0);
            cudaMemcpy(data, ones_data.data(), n * sizeof(double), cudaMemcpyHostToDevice);
            break;
        }
        case DataType::kInt32:
        {
            std::vector<int32_t> ones_data(n, 1);
            cudaMemcpy(data, ones_data.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
            break;
        }
        case DataType::kInt8:
        {
            std::vector<int8_t> ones_data(n, 1);
            cudaMemcpy(data, ones_data.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice);
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for CUDA ones operation", dtype_to_string(options.dtype()));
    }
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    return result;
}

/**
 * @brief 在CUDA设备上创建填充指定值的张量
 */
std::unique_ptr<OriginMat> full(const Shape &shape, double value, const TensorOptions &options)
{
    // 验证设备
    if (options.device().type() != DeviceType::kCUDA)
    {
        THROW_INVALID_ARG("CUDA full requires CUDA device, got: {}", options.device().to_string());
    }
    
    // 设置CUDA设备
    set_cuda_device(options.device().index());
    
    auto result = std::unique_ptr<OriginMat>(new OriginMat(shape, options.dtype(), options.device()));
    
    // 获取数据指针
    void *data = result->storage()->data();
    size_t n = shape.elements();
    
    // 根据数据类型设置值
    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            std::vector<float> fill_data(n, static_cast<float>(value));
            cudaMemcpy(data, fill_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            break;
        }
        case DataType::kFloat64:
        {
            std::vector<double> fill_data(n, value);
            cudaMemcpy(data, fill_data.data(), n * sizeof(double), cudaMemcpyHostToDevice);
            break;
        }
        case DataType::kInt32:
        {
            std::vector<int32_t> fill_data(n, static_cast<int32_t>(value));
            cudaMemcpy(data, fill_data.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
            break;
        }
        case DataType::kInt8:
        {
            std::vector<int8_t> fill_data(n, static_cast<int8_t>(value));
            cudaMemcpy(data, fill_data.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice);
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for CUDA full operation", dtype_to_string(options.dtype()));
    }
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    return result;
}

}  // namespace cuda
}  // namespace origin
