#include "originDL.h"

int main()
{
    // 创建标量
    double scalar_val = 1.0;
    dl::NdArray datax = af::constant(scalar_val, 1);
    auto x            = std::make_shared<dl::Variable>(datax);
    x->Print();

    // 创建一维数组
    double datay_data[] = {1.01, 2.01, 3.01};
    dl::NdArray datay = af::array(3, datay_data);
    auto y            = std::make_shared<dl::Variable>(datay);
    y->Print();

    // 创建二维数组
    double dataz_data[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0}; // 列优先存储
    dl::NdArray dataz = af::array(2, 3, dataz_data);
    auto z            = std::make_shared<dl::Variable>(dataz);
    z->Print();

    return 0;
}
