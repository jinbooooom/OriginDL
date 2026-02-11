#include <iostream>
#include "origin.h"

using namespace origin;

int main()
{
    // 标量，无维度
    {
        Tensor tensor(1.0, Shape{});
        tensor.print("tensor shape{}");
    }

    // 一维
    {
        Tensor tensor({1.01, 2.01, 3.01}, Shape{3}, Float32);
        tensor.print("tensor (3)");
    }
    {
        Tensor tensor({1.01, 2.01, 3.01}, Shape{1, 3}, Float32);
        tensor.print("tensor (1, 3)");
    }
    {
        Tensor tensor({1.01, 2.01, 3.01}, Shape{3, 1}, Float32);
        tensor.print("tensor (3, 1)");
    }

    // 二维
    {
        Tensor tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, Shape{3, 4}, Float32);
        tensor.print("tensor (3, 4)");
    }

    // 三维
    {
        Tensor tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, Shape{2, 3, 2}, Float32);
        tensor.print("tensor (2, 3, 2)");
    }

    // 四维
    {
        Shape shape = {2, 3, 2, 3};
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < shape.elements(); i++)
        {
            data[i] = i;
        }
        auto tensor = Tensor(data, shape, dtype(Float32).device(Device(kCPU)));
        tensor.print("tensor (2, 3, 2, 3)");
    }

    // 五维
    {
        Shape shape = {2, 3, 2, 3, 5};
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < shape.elements(); i++)
        {
            data[i] = i;
        }
        auto tensor = Tensor(data, shape, dtype(Float32).device(Device(kCPU)));
        tensor.print("tensor (2, 3, 2, 3, 5)");
    }

    // 测试大张量的省略功能
    {
        Shape shape = {100, 25};
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < shape.elements(); i++)
        {
            data[i] = i;
        }
        auto tensor = Tensor(data, shape, dtype(Float32).device(Device(kCPU)));
        tensor.print("tensor " + shape.to_string());
    }
    {
        Shape shape = {25, 100};
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < shape.elements(); i++)
        {
            data[i] = i;
        }
        auto tensor = Tensor(data, shape, dtype(Float32).device(Device(kCPU)));
        tensor.print("tensor " + shape.to_string());
    }

    return 0;
}
