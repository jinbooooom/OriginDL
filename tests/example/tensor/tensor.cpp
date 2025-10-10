#include <iostream>
#include "origin.h"

int main()
{
    // 创建标量
    origin::Tensor x(1.0, origin::Shape{1});
    x.print();

    // 创建一维数组
    std::vector<origin::data_t> datay_data = {1.01, 2.01, 3.01};
    origin::Tensor y(datay_data, origin::Shape{3});
    y.print();

    // 创建二维数组
    std::vector<origin::data_t> dataz_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    origin::Tensor z(dataz_data, origin::Shape{2, 3});
    z.print();

    return 0;
}
