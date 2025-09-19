



## 本仓库遵循的代码风格

### 代码格式化

使用 clang-format 进行代码格式化，配置文件为 `.clang-format`
python 命令：
```shell
python3 format.py
```

### 文件命名

```shell
# cpp 文件
myCppFile.cpp
myCppFile.h

# C 文件
myCFile.cc
```

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

统一使用小驼峰，不同的前缀含义如下：

- m：类内成员变量，注意结构体内的成员变量不加 m 前缀。
- g：全局变量
- s：静态变量
- k：常量

```cpp
int localVariable = 0;  // 普通变量
int mClassVariable = 1; // 类内成员变量加 m 前缀
int gGlobalVariable = 42; // 全局变量加 g 前缀
const int kDaysInAWeek = 7; // 常量
```

#### 函数命名

类内函数采用大驼峰，取值和设值函数则要求与变量名匹配: `MyExcitingFunction()`, `MyExcitingMethod()`, `my_exciting_member_variable()`, `set_my_exciting_member_variable()`.

```cpp
Class Animal
{
public:
      int Speak();
      void set_age(); // 设值的方法在成员变量前加上 set_
      int age(); // 取值的方法与成员变量一致，注意去掉了 m 前缀。
private:
     int mAge;
}
```

一般来说, 函数名的每个单词首字母大写, 没有下划线. 对于首字母缩写的单词（如简写RPC）, 更倾向于将它们视作一个单词进行首字母大写 (例如, 写作 `StartRpc()` 而非 `StartRPC()`).

#### 命名空间命名

命名空间以小写字母命，不使用缩写作为名称。

#### 枚举与宏

大写，且以下划线连接

```cpp
enum AlternateUrlTableErrors {
    OK = 0,
    OUT_OF_MEMORY = 1,
    MALFORMED_INPUT = 2,
}

#define ROUND(x) ...
#define PI_ROUNDED (3.0)
```

### 参考

本项目的编程风格部分参考 google style

- https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/naming.html
- https://github.com/zh-google-styleguide/zh-google-styleguide