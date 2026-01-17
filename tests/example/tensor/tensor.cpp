#include <iostream>
#include "origin.h"

using namespace origin;

int main()
{
    // 创建标量
    Tensor t1(1.0, Shape{1});
    t1.print("t1");

    // 创建一维数组
    std::vector<double> d2 = {1.01, 2.01, 3.01};
    Tensor t2(d2, Shape{3}, dtype("int32"));
    t2.print("t2");

    // 创建二维数组
    std::vector<data_t> d3 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor t3(d3, Shape{2, 3});
    t3.print("t3");

    Tensor t4({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3}, Float32);
    t4.print("t4");

    std::vector<double> data5 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor t5(data5, Shape{3, 2}, dtype(Float32));
    t5.print("t5");

    auto t6 = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(Device(DeviceType::kCPU)));
    t6.print("t6");

    return 0;
}
