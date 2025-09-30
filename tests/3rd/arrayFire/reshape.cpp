#include <arrayfire.h>

int main()
{
    // 初始化 ArrayFire
    af::setDevice(0);
    af::info();

    // 创建示例张量 (3x4)
    af::array tensor = af::randu(3, 4);  // 3行4列随机值
    af::print("Original tensor (3x4):", tensor);

    // ----------------------------
    // 1. 重塑维度 (reshape)
    // ----------------------------
    af::dim4 shape4_3{4, 3};
    af::array reshaped = af::moddims(tensor, shape4_3);  // 改为4x3
    af::print("\nReshaped to 4x3:", reshaped);

    // 展平为一维
    af::array flattened = af::flat(tensor);
    af::print("\nFlattened (1D):", flattened);

    // ----------------------------
    // 2. 维度重排 (reorder)
    // ----------------------------
    // 创建3D张量 (2x3x2)
    af::array tensor3d = af::randu(2, 3, 2);
    af::print("\nOriginal 3D tensor (2x3x2):", tensor3d);

    // 重排维度 [0,1,2] -> [2,0,1]
    af::array reordered = af::reorder(tensor3d, 2, 0, 1);
    af::print("\nReordered (2->0, 0->1, 1->2):", reordered);

    // ----------------------------
    // 3. 转置 (transpose)
    // ----------------------------
    af::array transposed = af::transpose(tensor);  // 行列交换
    af::print("\nTransposed (4x3):", transposed);

    // ----------------------------
    // 4. 切片与子张量操作
    // ----------------------------
    // 取第1行
    af::array row_slice = tensor.row(0);
    af::print("\nFirst row slice:", row_slice);

    // 取第2-3列
    af::array col_slice = tensor.cols(1, 2);
    af::print("\nColumns 1-2 slice:", col_slice);

    // 修改子张量
    tensor.row(0) = af::constant(1, 1, 4);  // 第一行全赋值为1
    af::print("\nAfter modifying row 0:", tensor);

    // ----------------------------
    // 5. 数据类型转换 (配合shape操作)
    // ----------------------------
    af::array float_tensor = tensor.as(f32);  // 转为float32
    af::array half_tensor  = tensor.as(f16);  // 转为float16

    return 0;
}
