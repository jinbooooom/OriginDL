#include <iostream>
#include "originDL.h"

int main()
{
    // 初始化ArrayFire后端
    try
    {
        af::setBackend(AF_BACKEND_CPU);  // 使用CPU后端
        af::info();
        printf("\n");  // 输出设备信息并初始化ArrayFire
    }
    catch (const af::exception &e)
    {
        std::cerr << "Failed to initialize ArrayFire: " << e.what() << std::endl;
        return 1;
    }

    // 创建标量
    dl::Tensor x(1.0, dl::Shape{1});
    x.print();

    // 创建一维数组
    std::vector<dl::data_t> datay_data = {1.01, 2.01, 3.01};
    dl::Tensor y(datay_data, dl::Shape{3});
    y.print();

    // 创建二维数组
    std::vector<dl::data_t> dataz_data = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};  // 列优先存储
    dl::Tensor z(dataz_data, dl::Shape{2, 3});
    z.print();

    return 0;
}
