#include <iostream>
#include "origin.h"

int main()
{
    // 标量，无维度
    {
        origin::Tensor tensor(1.0, origin::Shape{});
        tensor.print("tensor shape{}");
    }

    // 一维
    {
        origin::Tensor tensor({1.01, 2.01, 3.01}, origin::Shape{3}, origin::Float32);
        tensor.print("tensor (3)");
    }
    {
        origin::Tensor tensor({1.01, 2.01, 3.01}, origin::Shape{1, 3}, origin::Float32);
        tensor.print("tensor (1, 3)");
    }
    {
        origin::Tensor tensor({1.01, 2.01, 3.01}, origin::Shape{3, 1}, origin::Float32);
        tensor.print("tensor (3, 1)");
    }

    // 二维
    {
        origin::Tensor tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, origin::Shape{3, 4}, origin::Float32);
        tensor.print("tensor (3, 4)");
    }

    // 三维
    {
        origin::Tensor tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, origin::Shape{2, 3, 2}, origin::Float32);
        tensor.print("tensor (2, 3, 2)");
    }

    // 四维
    {
        origin::Shape shape = {2, 3, 2, 3};
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < shape.elements(); i++)
        {
            data[i] = i;
        }
        auto tensor = origin::Tensor(data, shape, origin::dtype(origin::Float32).device(origin::Device(origin::kCPU)));
        tensor.print("tensor (2, 3, 2, 3)");
    }

    // 五维
    {
        origin::Shape shape = {2, 3, 2, 3, 5};
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < shape.elements(); i++)
        {
            data[i] = i;
        }
        auto tensor = origin::Tensor(data, shape, origin::dtype(origin::Float32).device(origin::Device(origin::kCPU)));
        tensor.print("tensor (2, 3, 2, 3, 5)");
    }

    // 测试大张量的省略功能
    {
        origin::Shape shape = {100, 25};
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < shape.elements(); i++)
        {
            data[i] = i;
        }
        auto tensor = origin::Tensor(data, shape, origin::dtype(origin::Float32).device(origin::Device(origin::kCPU)));
        tensor.print("tensor " + shape.to_string());
    }
    {
        origin::Shape shape = {25, 100};
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < shape.elements(); i++)
        {
            data[i] = i;
        }
        auto tensor = origin::Tensor(data, shape, origin::dtype(origin::Float32).device(origin::Device(origin::kCPU)));
        tensor.print("tensor " + shape.to_string());
    }

    return 0;
}
