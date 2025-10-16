#include <iostream>
#include "origin/core/tensor.h"
#include "origin/core/operator.h"
#include "origin/mat/origin/cuda/cuda_utils.cuh"

using namespace origin;

int main()
{
    std::cout << "Testing minimal CUDA operations..." << std::endl;
    
    // 检查CUDA是否可用
    if (!cuda::is_cuda_available()) {
        std::cout << "CUDA not available!" << std::endl;
        return 1;
    }
    
    try {
        // 创建简单的张量
        std::cout << "Creating tensors..." << std::endl;
        auto a = Tensor::ones({2}, TensorOptions().dtype(DataType::kFloat32).device(Device(DeviceType::kCUDA)));
        auto b = Tensor::ones({2}, TensorOptions().dtype(DataType::kFloat32).device(Device(DeviceType::kCUDA)));
        
        std::cout << "A created successfully" << std::endl;
        std::cout << "B created successfully" << std::endl;
        
        // 测试加法
        std::cout << "Testing addition..." << std::endl;
        auto c = a + b;
        std::cout << "Addition completed successfully" << std::endl;
        
        // 测试乘法
        std::cout << "Testing multiplication..." << std::endl;
        auto d = a * b;
        std::cout << "Multiplication completed successfully" << std::endl;
        
        std::cout << "All tests passed!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
