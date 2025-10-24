#include <iostream>
#include "origin.h"

// using namespace origin;

int main()
{
    // 创建标量
    origin::Tensor t1(1.0, origin::Shape{1});
    t1.print("t1");

    // 创建一维数组
    std::vector<double> d2 = {1.01, 2.01, 3.01};
    origin::Tensor t2(d2, origin::Shape{3}, origin::dtype("int32"));
    t2.print("t2");

    // 创建二维数组
    std::vector<origin::data_t> d3 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    origin::Tensor t3(d3, origin::Shape{2, 3});
    t3.print("t3");

    origin::Tensor t4({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, origin::Shape{2, 3}, origin::Float32);
    t4.print("t4");

    std::vector<double> data5 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    origin::Tensor t5(data5, origin::Shape{3, 2}, origin::dtype(origin::Float32));
    t5.print("t5");

    auto t6 = origin::Tensor::ones(
        origin::Shape{2, 2},
        origin::dtype(origin::DataType::kFloat32).device(origin::Device(origin::DeviceType::kCPU)));
    t6.print("t6");

    return 0;
}
