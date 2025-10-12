#include "origin.h"
#include "origin/utils/log.h"

using namespace origin;

int main()
{

    data_t val0     = 2;
    data_t val1     = 4;
    Shape shape     = {2, 2};
    auto x0         = Tensor(val0, shape);
    auto x1         = Tensor(val1, shape);
    auto y          = -x0;
    auto clear_grad = [&]() {
        y.clear_grad();
        x0.clear_grad();
        x1.clear_grad();
    };

    logi("Neg: y = -x0");
    clear_grad();
    y = -x0;
    y.backward();
    y.print("y ");
    x0.grad().print("dx0 ");

    logi("Add: y = x0 + x1");
    clear_grad();
    y = x0 + x1;
    y.backward();
    y.print("y ");
    x0.grad().print("dx0 ");
    x1.grad().print("dx1 ");

    logi("Sub: y = x0 - x1");
    clear_grad();
    y = x0 - x1;
    y.backward();
    y.print("y ");
    x0.grad().print("dx0 ");
    x1.grad().print("dx1 ");

    logi("Mul: y = x0 * x1");
    clear_grad();
    y = x0 * x1;
    y.backward();
    y.print("y ");
    x0.grad().print("dx0 ");
    x1.grad().print("dx1 ");

    logi("Div: y = x0 / x1");
    clear_grad();
    y = x0 / x1;
    y.backward();
    y.print("y ");
    x0.grad().print("dx0 ");  // 1 / x1
    x1.grad().print("dx1 ");  // -x0 / x1^2

    logi("Square: y = x0^2");
    clear_grad();
    y = square(x0);
    y.backward();
    y.print("y ");
    x0.grad().print("dx0 ");

    logi("Pow: y = x0^3");
    clear_grad();
    y = x0 ^ 3;
    y.backward();
    y.print("y ");
    x0.grad().print("dx0 ");

    logi("Exp: y = exp(x0)");
    clear_grad();
    y = exp(x0);
    y.backward();
    y.print("y ");
    x0.grad().print("dx0 ");

    // 测试Reshape算子
    auto x = Tensor({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, Shape{2, 4}, DataType::kFloat32);
    x.print("x ");
    logi("Reshape: y = reshape(x, {4, 2})");
    x.clear_grad();
    auto x_reshaped = reshape(x, Shape{4, 2});
    x_reshaped.backward();
    x_reshaped.print("y ");
    x.grad().print("dx ");

    // 测试Transpose算子
    logi("Transpose: y = transpose(x)");
    x.clear_grad();
    auto x_transposed = transpose(x);
    x_transposed.backward();
    x_transposed.print("y ");
    x.grad().print("dx ");

    // 测试Sum算子
    logi("Sum: y = sum(x)");
    x.clear_grad();
    auto x_sum = sum(x);
    x_sum.backward();
    x_sum.print("y ");
    x.grad().print("dx ");

    // 测试BroadcastTo算子 - 使用更合适的广播用例
    // 注意：LibTorch的广播规则要求从右到左比较维度，每个维度要么大小相同，要么其中一个为1，要么其中一个不存在
    // 错误的广播示例：[2,4] -> [2,4,4] 或 [2,4] -> [2,4,1] 会失败，因为第2个维度不存在于源tensor中
    // 正确的广播示例：[2,4] -> [2,4] 或 [1,4] -> [2,4] 或 [2,1] -> [2,4]
    logi("BroadcastTo: y = broadcast_to(x, {2, 4})");
    x.clear_grad();
    auto x_broadcasted = broadcast_to(x, Shape{2, 4});
    x_broadcasted.backward();
    x_broadcasted.print("y ");
    x.grad().print("dx ");

    // 测试SumTo算子
    logi("SumTo: y = sum_to(x, {1, 1})");
    x.clear_grad();
    auto x_sum_to = sum_to(x, Shape{1, 1});
    x_sum_to.backward();
    x_sum_to.print("y ");
    x.grad().print("dx ");

    // 测试MatMul算子 - 使用简单的2x2矩阵
    auto a = Tensor({1, 2, 3, 4}, Shape{2, 2});
    auto b = Tensor({5, 6, 7, 8}, Shape{2, 2});
    logi("MatMul: y = mat_mul(a, b)");
    a.clear_grad();
    b.clear_grad();
    auto ab_matmul = mat_mul(a, b);
    ab_matmul.backward();
    ab_matmul.print("y ");
    a.grad().print("da ");
    b.grad().print("db ");

    return 0;
}
