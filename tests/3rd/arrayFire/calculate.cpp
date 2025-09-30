#include <arrayfire.h>
#include <iostream>

int main()
{
    try
    {
        // 初始化矩阵（列优先存储）
        float a_data[] = {1, 3, 2, 4};  // 矩阵A [[1,2],[3,4]]
        float b_data[] = {1, 0, 0, 1};  // 矩阵B [[1,0],[0,1]]
        af::array A    = af::array(2, 2, a_data);
        af::array B    = af::array(2, 2, b_data);

        // 指数运算（逐元素）
        af::array expA = af::exp(A);

        // 平方运算（逐元素）
        af::array squareA = af::pow(A, 2);

        // 逐元素乘
        af::array AA = A * A;

        // 矩阵加法
        af::array addAB = A + B;

        // 矩阵乘法（行列式乘积）
        af::array matmulAB = af::matmul(A, B);

        // 点积运算（Frobenius内积）
        af::array dotAB = af::dot(flat(A), flat(B));  // flat(A) 变为 4x1 向量

        // 结果输出
        af::print("Matrix A", A);
        af::print("Matrix B", B);
        af::print("exp(A)", expA);
        af::print("A squared", squareA);
        af::print("A * A", squareA);
        af::print("A + B", addAB);
        af::print("A * B (Matrix multiply)", matmulAB);
        af::print("A dot B", dotAB);
    }
    catch (const af::exception &e)
    {
        std::cerr << "ArrayFire error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

/*
$ g++ calculate.cpp  -o ./bin/calculate -I/opt/arrayfire/include -L/opt/arrayfire/lib64  -laf
$ ./bin/calculate
Matrix A
[2 2 1 1]
    1.0000     2.0000
    3.0000     4.0000

Matrix B
[2 2 1 1]
    1.0000     0.0000
    0.0000     1.0000

exp(A)
[2 2 1 1]
    2.7183     7.3891
   20.0855    54.5981

A squared
[2 2 1 1]
    1.0000     4.0000
    9.0000    16.0000

A * A
[2 2 1 1]
    1.0000     4.0000
    9.0000    16.0000

A + B
[2 2 1 1]
    2.0000     2.0000
    3.0000     5.0000

A * B (Matrix multiply)
[2 2 1 1]
    1.0000     2.0000
    3.0000     4.0000

A dot B
[1 1 1 1]
    5.0000
*/