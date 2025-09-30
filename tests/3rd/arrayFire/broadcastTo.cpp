#include <arrayfire.h>
#include <iostream>
#include <vector>

// 自定义 broadcast_to 函数
af::array broadcast_to(const af::array &src, const af::dim4 &target_shape)
{
    // 获取源数组和目标形状的维度
    af::dim4 src_dims = src.dims();

    // 如果源数组已经是目标形状，则直接返回
    if (src_dims == target_shape)
    {
        return src;
    }

    // 处理标量或单元素数组的特殊情况
    if (src.elements() == 1)
    {
        // 如果是标量，直接创建目标形状的常量数组
        float value;
        src.host(&value);
        return af::constant(value, target_shape);
    }

    // 对于一般情况，我们需要逐维度处理
    // 首先确保源维度与目标维度兼容
    for (int i = 0; i < 4; i++)
    {
        if (src_dims[i] != 1 && src_dims[i] != target_shape[i])
        {
            throw std::runtime_error("Incompatible dimensions for broadcasting");
        }
    }

    // 计算每个维度上需要重复的次数
    unsigned repeat_0 = (src_dims[0] == 1) ? target_shape[0] : 1;
    unsigned repeat_1 = (src_dims[1] == 1) ? target_shape[1] : 1;
    unsigned repeat_2 = (src_dims[2] == 1) ? target_shape[2] : 1;
    unsigned repeat_3 = (src_dims[3] == 1) ? target_shape[3] : 1;

    // 使用 tile 函数进行广播
    return af::tile(src, repeat_0, repeat_1, repeat_2, repeat_3);
}

// 演示 ArrayFire 的隐式广播
void demo_implicit_broadcasting()
{
    std::cout << "\n=== 隐式广播演示 ===" << std::endl;

    // 创建一个 1x3 数组
    float data1[] = {1, 2, 3};
    af::array a(1, 3, data1);
    std::cout << "数组 a (1x3):" << std::endl;
    af::print("a", a);

    // 创建一个 2x1 数组
    float data2[] = {10, 20};
    af::array b(2, 1, data2);
    std::cout << "数组 b (2x1):" << std::endl;
    af::print("b", b);

    // 使用算术运算符进行隐式广播
    af::array c = a + b;  // 结果将是 2x3
    std::cout << "a + b (隐式广播到 2x3):" << std::endl;
    af::print("c", c);
}

// 演示自定义的 broadcast_to 函数
void demo_custom_broadcast_to()
{
    std::cout << "\n=== 自定义 broadcast_to 函数演示 ===" << std::endl;

    // 创建一个标量数组
    af::array scalar = af::constant(5, 1);
    std::cout << "标量:" << std::endl;
    af::print("scalar", scalar);

    // 广播到 2x3 形状
    af::dim4 target_shape(2, 3);
    af::array broadcasted = broadcast_to(scalar, target_shape);
    std::cout << "广播到 2x3:" << std::endl;
    af::print("broadcasted", broadcasted);

    // 创建一个 1x3 数组
    float data[] = {1, 2, 3};
    af::array a(1, 3, data);
    std::cout << "数组 a (1x3):" << std::endl;
    af::print("a", a);

    // 广播到 2x3 形状
    af::array broadcasted_a = broadcast_to(a, af::dim4(2, 3));
    std::cout << "a 广播到 2x3:" << std::endl;
    af::print("broadcasted_a", broadcasted_a);

    // 创建一个 2x1 数组
    float data2[] = {10, 20};
    af::array b(2, 1, data2);
    std::cout << "数组 b (2x1):" << std::endl;
    af::print("b", b);

    // 广播到 2x3 形状
    af::array broadcasted_b = broadcast_to(b, af::dim4(2, 3));
    std::cout << "b 广播到 2x3:" << std::endl;
    af::print("broadcasted_b", broadcasted_b);
}

// 演示 af::tile 函数
void demo_tile()
{
    std::cout << "\n=== af::tile 函数演示 ===" << std::endl;

    // 创建一个 1x3 数组
    float data[] = {1, 2, 3};
    af::array a(1, 3, data);
    std::cout << "数组 a (1x3):" << std::endl;
    af::print("a", a);

    // 使用 tile 在第一个维度上重复2次
    af::array tiled_0 = af::tile(a, 2);
    std::cout << "a 在第一个维度上重复2次:" << std::endl;
    af::print("tiled_0", tiled_0);

    // 使用 tile 在第二个维度上重复2次
    af::array tiled_1 = af::tile(a, 1, 2);
    std::cout << "a 在第二个维度上重复2次:" << std::endl;
    af::print("tiled_1", tiled_1);
}

// 演示 sum 操作的反向传播
void demo_sum_backward()
{
    std::cout << "\n=== Sum 反向传播演示 ===" << std::endl;

    // 创建一个 2x3 数组作为输入
    af::array x = af::randu(2, 3);
    std::cout << "输入 x (2x3):" << std::endl;
    af::print("x", x);

    // 沿第0维度求和 (结果是 1x3)
    af::array y_0 = af::sum(x, 0);
    std::cout << "y_0 = sum(x, 0) (1x3):" << std::endl;
    af::print("y_0", y_0);

    // 假设梯度是 1
    af::array gy_0 = af::constant(1, y_0.dims());
    std::cout << "梯度 gy_0 (1x3):" << std::endl;
    af::print("gy_0", gy_0);

    // 反向传播 - 广播到原始形状
    af::array gx_0 = broadcast_to(gy_0, x.dims());
    std::cout << "反向传播梯度 gx_0 (2x3):" << std::endl;
    af::print("gx_0", gx_0);

    // 沿第1维度求和 (结果是 2x1)
    af::array y_1 = af::sum(x, 1);
    std::cout << "y_1 = sum(x, 1) (2x1):" << std::endl;
    af::print("y_1", y_1);

    // 假设梯度是 1
    af::array gy_1 = af::constant(1, y_1.dims());
    std::cout << "梯度 gy_1 (2x1):" << std::endl;
    af::print("gy_1", gy_1);

    // 反向传播 - 广播到原始形状
    af::array gx_1 = broadcast_to(gy_1, x.dims());
    std::cout << "反向传播梯度 gx_1 (2x3):" << std::endl;
    af::print("gx_1", gx_1);

    // 对所有元素求和 (结果是标量)
    af::array y_all = af::sum(af::sum(x, 0), 1);
    std::cout << "y_all = sum(sum(x, 0), 1) (标量):" << std::endl;
    af::print("y_all", y_all);

    // 假设梯度是 1
    af::array gy_all = af::constant(1, 1);
    std::cout << "梯度 gy_all (标量):" << std::endl;
    af::print("gy_all", gy_all);

    // 反向传播 - 广播到原始形状
    af::array gx_all = broadcast_to(gy_all, x.dims());
    std::cout << "反向传播梯度 gx_all (2x3):" << std::endl;
    af::print("gx_all", gx_all);
}

int main()
{
    try
    {
        // 设置设备
        af::info();

        // 演示隐式广播
        demo_implicit_broadcasting();

        // 演示自定义 broadcast_to 函数
        demo_custom_broadcast_to();

        // 演示 tile 函数
        demo_tile();

        // 演示 sum 操作的反向传播
        demo_sum_backward();
    }
    catch (af::exception &e)
    {
        std::cerr << "ArrayFire 异常: " << e.what() << std::endl;
        return -1;
    }
    catch (std::exception &e)
    {
        std::cerr << "标准异常: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

/*
$ g++ broadcastTo.cpp -o broadcastTo  -I/opt/arrayfire/include -L/opt/arrayfire/lib64  -laf
$ ./broadcastTo
ArrayFire v3.9.0 (CPU, 64-bit Linux, build b59a1ae53)
[0] Intel: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
=== 隐式广播演示 ===
数组 a (1x3):
a
[1 3 1 1]
    1.0000     2.0000     3.0000
数组 b (2x1):
b
[2 1 1 1]
   10.0000
   20.0000

a + b (隐式广播到 2x3):
c
[2 3 1 1]
   11.0000    12.0000    13.0000
   21.0000    22.0000    23.0000


=== 自定义 broadcast_to 函数演示 ===
标量:
scalar
[1 1 1 1]
    5.0000
广播到 2x3:
broadcasted
[2 3 1 1]
    5.0000     5.0000     5.0000
    5.0000     5.0000     5.0000

数组 a (1x3):
a
[1 3 1 1]
    1.0000     2.0000     3.0000
a 广播到 2x3:
broadcasted_a
[2 3 1 1]
    1.0000     2.0000     3.0000
    1.0000     2.0000     3.0000

数组 b (2x1):
b
[2 1 1 1]
   10.0000
   20.0000

b 广播到 2x3:
broadcasted_b
[2 3 1 1]
   10.0000    10.0000    10.0000
   20.0000    20.0000    20.0000


=== af::tile 函数演示 ===
数组 a (1x3):
a
[1 3 1 1]
    1.0000     2.0000     3.0000
a 在第一个维度上重复2次:
tiled_0
[2 3 1 1]
    1.0000     2.0000     3.0000
    1.0000     2.0000     3.0000

a 在第二个维度上重复2次:
tiled_1
[1 6 1 1]
    1.0000     2.0000     3.0000     1.0000     2.0000     3.0000

=== Sum 反向传播演示 ===
输入 x (2x3):
x
[2 3 1 1]
    0.6010     0.9806     0.0655
    0.0278     0.2126     0.5497

y_0 = sum(x, 0) (1x3):
y_0
[1 3 1 1]
    0.6287     1.1932     0.6151
梯度 gy_0 (1x3):
gy_0
[1 3 1 1]
    1.0000     1.0000     1.0000
反向传播梯度 gx_0 (2x3):
gx_0
[2 3 1 1]
    1.0000     1.0000     1.0000
    1.0000     1.0000     1.0000

y_1 = sum(x, 1) (2x1):
y_1
[2 1 1 1]
    1.6470
    0.7901

梯度 gy_1 (2x1):
gy_1
[2 1 1 1]
    1.0000
    1.0000

反向传播梯度 gx_1 (2x3):
gx_1
[2 3 1 1]
    1.0000     1.0000     1.0000
    1.0000     1.0000     1.0000

y_all = sum(sum(x, 0), 1) (标量):
y_all
[1 1 1 1]
    2.4370
梯度 gy_all (标量):
gy_all
[1 1 1 1]
    1.0000
反向传播梯度 gx_all (2x3):
gx_all
[2 3 1 1]
    1.0000     1.0000     1.0000
    1.0000     1.0000     1.0000
*/
