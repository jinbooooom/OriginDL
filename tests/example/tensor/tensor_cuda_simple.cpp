#include <iostream>
#include "origin.h"
namespace F = origin::functional;

int main()
{
    origin::cuda::device_info();

    if (!origin::cuda::is_available())
    {
        std::cout << "CUDA is not available on this system!" << std::endl;
        return 1;
    }

    std::cout << "Testing simple CUDA operations..." << std::endl;

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

    // Test multiplication
    {
        std::cout << "\n--- Testing Multiplication ---" << std::endl;
        origin::Tensor a({2.0f, 3.0f}, origin::Shape{2}, origin::dtype(origin::Float32).device(origin::kCUDA));
        origin::Tensor b({4.0f, 5.0f}, origin::Shape{2}, origin::dtype(origin::Float32).device(origin::kCUDA));

        a.print("A");
        b.print("B");

        auto c = a * b;
        c.print("A * B");
    }

    std::cout << "All tests completed successfully!" << std::endl;

    return 0;
}
