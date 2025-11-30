#include <iostream>
#include "origin.h"

int main()
{
#ifdef WITH_CUDA
    origin::cuda::print_cuda_device_info();

    if (!origin::cuda::is_cuda_available())
    {
        std::cout << "CUDA is not available on this system!" << std::endl;
        return 1;
    }

    std::cout << "Testing basic CUDA operations..." << std::endl;

    // Test basic addition first
    {
        std::cout << "\n--- Testing Basic Addition ---" << std::endl;
        origin::Tensor a({1.0f, 2.0f}, origin::Shape{2}, origin::dtype(origin::Float32).device(origin::kCUDA));
        origin::Tensor b({3.0f, 4.0f}, origin::Shape{2}, origin::dtype(origin::Float32).device(origin::kCUDA));

        a.print("A");
        b.print("B");

        auto c = a + b;
        c.print("A + B");
    }

    // Test subtraction
    {
        std::cout << "\n--- Testing Subtraction ---" << std::endl;
        origin::Tensor a({5.0f, 3.0f}, origin::Shape{2}, origin::dtype(origin::Float32).device(origin::kCUDA));
        origin::Tensor b({2.0f, 1.0f}, origin::Shape{2}, origin::dtype(origin::Float32).device(origin::kCUDA));

        a.print("A");
        b.print("B");

        auto c = a - b;
        c.print("A - B");
    }

    std::cout << "All tests completed successfully!" << std::endl;

#else
    std::cout << "CUDA support is not enabled in this build!" << std::endl;
    std::cout << "Please rebuild with --cuda flag: ./build.sh ORIGIN --cuda" << std::endl;
    return 1;
#endif

    return 0;
}
