



## 本仓库遵循的代码风格

### 代码格式化

使用 clang-format 进行代码格式化，配置文件为 `.clang-format`
python 命令：

```shell
python3 format.py
```

### 文件命名

通常应尽量让文件名更加明确. `http_server_logs.h` 就比 `logs.h` 要好. 定义类时文件名一般成对出现, 如 `foo_bar.h` 和 `foo_bar.cpp`, 对应于类 `FooBar`。

### 重复include保护

使用 #define 宏 进行 .h 重复include保护，不使用 \#pragma once

```c++
#ifndef __THIS_IS_A_HEADER_H__
#define __THIS_IS_A_HEADER_H__ 
#endif // __THIS_IS_A_HEADER_H__ 
```

### 命名规则

#### 类型命名

所有类型命名 —— 类, 结构体, 类型定义 (`typedef`), 枚举, 类型模板参数 —— 均使用相同约定, 即以大写字母开始, 每个单词首字母均大写, 不包含下划线. 例如:

```cpp
// 类和结构体
class UrlTable { ...
class UrlTableTester { ...
struct UrlTableProperties { ...

// 类型定义
typedef hash_map<UrlTableProperties *, string> PropertiesMap;

// using 别名
using PropertiesMap = hash_map<UrlTableProperties *, string>;

// 枚举
enum UrlTableErrors { ...
```

#### 变量命名

统一使用小写字母和下划线分隔 (snake_case)，不同的前缀含义如下：

- 无前缀：普通局部变量和成员变量
- g_：全局变量
- s_：静态变量
- k：常量

```cpp
int local_variable = 0;        // 普通变量
int batch_size = 32;           // 成员变量
int g_global_variable = 42;    // 全局变量加 g_ 前缀
static int s_static_var = 1;   // 静态变量加 s_ 前缀
const int kMaxEpochs = 100;     // 常量
```

##### 类数据成员

不管是静态的还是非静态的, 类数据成员都可以和普通变量一样, 但要接下划线.

```cpp
class TableInfo {
  ...
 private:
  string table_name_;  // 后加下划线.
  static Pool<TableInfo>* pool_;  // 后加下划线.
};
```

##### 结构体变量

不管是静态的还是非静态的, 结构体数据成员都可以和普通变量一样, 不用像类那样接下划线:

```cpp
struct UrlTableProperties {
  string name;
  int num_entries;
  static Pool<UrlTableProperties>* pool;
};
```

##### 常量命名

声明为 `constexpr` 或 `const` 的变量, 或在程序运行期间其值始终保持不变的, 命名时以 “k” 开头, 大小写混合. 例如:

```cpp
const int kDaysInAWeek = 7;
```

#### 函数命名

使用小写字母和下划线分隔 (snake_case)，需要注意的是设置值的函数需要加 set_ 前缀，但是取值函数名与变量名相同。

```cpp
class DataLoader
{
public:
    void load_data();                    // 普通函数
    void set_batch_size(int size);      // 设值函数
    int batch_size() const;         // 取值函数，不加 set，函数名就是变量名。
    Tensor forward();            // 前向传播
    void backward();                    // 反向传播
private:
    int batch_size_;
    std::string dataset_path_;
};
```

#### 枚举与宏

大写，且以下划线连接

```cpp
enum AlternateUrlTableErrors {
    OK = 0,
    OUT_OF_MEMORY = 1,
    MALFORMED_INPUT = 2,
}

#define MAX_BATCH_SIZE (1024)
#define PI_ROUNDED (3.14159)
```

### 采用异常而非错误码
pytorch 内部使用异常，它的python接口基于pybind11，pybind11会负责将 cpp 异常转换为 python 异常。本仓库将采用类似的方式。

### 注释规范

使用 Doxygen 风格的注释：

```cpp
/**
 * @brief 数据加载器类，用于加载和预处理数据集
 * @details 该类支持多种数据格式的加载，包括图像、文本等
 */
class DataLoader {
public:
    /**
     * @brief 构造函数
     * @param dataset_path 数据集路径
     * @param batch_size 批次大小，默认为32
     * @param shuffle 是否打乱数据，默认为true
     */
    explicit DataLoader(const std::string& dataset_path, 
                       int batch_size = 32, 
                       bool shuffle = true);
    
    /**
     * @brief 加载下一批次数据
     * @return 包含输入和标签的批次数据
     * @throws DataLoadException 当数据加载失败时抛出
     */
    std::pair<torch::Tensor, torch::Tensor> next_batch();
    
private:
    std::string dataset_path_;  ///< 数据集路径
    int batch_size_;           ///< 批次大小
    bool shuffle_;             ///< 是否打乱数据
};
```

### 参考

本项目的编程风格部分参考 google style 以及 pytorch C++ 代码风格

- https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/naming.html
- https://github.com/zh-google-styleguide/zh-google-styleguide