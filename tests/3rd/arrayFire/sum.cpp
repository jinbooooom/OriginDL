#include <arrayfire.h>
#include <iostream>

int main()
{
    // 初始化 ArrayFire 环境
    af::setDevice(0);
    af::info();

    // 创建示例矩阵 (3x4)
    af::array A = af::iota(af::dim4(3, 4));  // 生成 [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
    af::print("Original Matrix A:", A);

    // 按轴求和操作
    // 1. 按行求和 (axis=0)：每列元素相加 → 结果形状 (1x4)
    af::array sum_axis0 = af::sum(A, 0);
    af::print("Sum along axis=0 (rows):", sum_axis0);  // 输出: [12, 15, 18, 21]

    // 2. 按列求和 (axis=1)：每行元素相加 → 结果形状 (3x1)
    af::array sum_axis1 = af::sum(A, 1);
    af::print("Sum along axis=1 (columns):", sum_axis1);  // 输出: [6, 22, 38]

    // 3. 全局求和（无指定轴则是按列求和，和 numpy 不一样，numpy 直接逐元素求和）
    /* 嵌套求和的数学本质：逐层降维
    第一层 af::sum(A)：
    沿行方向（维度 0）压缩，消除行维度，生成行向量（如 1×4）。
    第二层 af::sum(row_vector)：
    对行向量再次求和。由于行向量只有一维（可视为 1×n 矩阵），沿其唯一的列方向（维度 0）压缩，最终输出标量（1×1）
    */
    af::array total_sum = af::sum(af::sum(A));
    af::print("Total sum (no axis):", total_sum);  // 输出: 66

    // 如果维度大于2维，就不能使用 af::sum(af::sum(A)) 了，需要对矩阵的每个维度进行求和
    // 创建示例矩阵 (3*4*2*2)
    af::array AA = af::iota(af::dim4(3, 4, 2, 2));
    auto n       = AA.numdims();
    printf("numdims of matrix AA: %d\n", n);
    af::array total_sum_AA = AA;
    for (int i = 0; i < n; ++i)
    {
        total_sum_AA = af::sum(total_sum_AA);
    }
    af::print("Total sum of AA(3*4*2*2):", total_sum_AA);  // 输出: 1128 = (0 + 47) * 48 / 2

    // 4. 结合条件求和
    af::array mask = (A > 5);
    af::print("Mask:", mask);
    af::print("A > 5:", A * mask);
    af::array conditional_sum = af::sum(A * mask, 1);  // 每行中 >5 的元素求和
    af::print("Conditional sum:", conditional_sum);

    return 0;
}

/*
g++ sum.cpp -o sum  -I/opt/arrayfire/include -L/opt/arrayfire/lib64  -laf
*/
