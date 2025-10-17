// 适合OriginDL的改进Scalar实现
namespace origin {

class Scalar {
private:
    // 使用对齐的union确保内存安全
    union alignas(8) {
        float f;       // 单精度浮点数
        double d;      // 双精度浮点数
        int32_t i;     // 32位整数
        int64_t l;     // 64位整数
        bool b;        // 布尔值
    } v;
    DataType type_;    // 使用OriginDL现有的DataType
    
public:
    // 构造函数 - 支持隐式转换
    Scalar(float f) : v{f}, type_(DataType::kFloat32) {}
    Scalar(double d) : v{d}, type_(DataType::kFloat64) {}
    Scalar(int32_t i) : v{i}, type_(DataType::kInt32) {}
    Scalar(int64_t l) : v{l}, type_(DataType::kInt64) {}  // 需要添加kInt64
    Scalar(bool b) : v{b}, type_(DataType::kBool) {}      // 需要添加kBool
    

    // 默认构造函数
    Scalar() : v{0.0f}, type_(DataType::kFloat32) {}
    
    // 类型转换 - 带错误检查
    float toFloat() const {
        if (type_ == DataType::kFloat32) return v.f;
        if (type_ == DataType::kFloat64) return static_cast<float>(v.d);
        if (type_ == DataType::kInt32) return static_cast<float>(v.i);
        if (type_ == DataType::kInt64) return static_cast<float>(v.l);
        if (type_ == DataType::kBool) return v.b ? 1.0f : 0.0f;
        THROW_INVALID_ARG("Cannot convert Scalar to float from type: {}", static_cast<int>(type_));
    }
    
    double toDouble() const {
        if (type_ == DataType::kFloat32) return static_cast<double>(v.f);
        if (type_ == DataType::kFloat64) return v.d;
        if (type_ == DataType::kInt32) return static_cast<double>(v.i);
        if (type_ == DataType::kInt64) return static_cast<double>(v.l);
        if (type_ == DataType::kBool) return v.b ? 1.0 : 0.0;
        THROW_INVALID_ARG("Cannot convert Scalar to double from type: {}", static_cast<int>(type_));
    }
    
    int32_t toInt32() const {
        if (type_ == DataType::kFloat32) return static_cast<int32_t>(v.f);
        if (type_ == DataType::kFloat64) return static_cast<int32_t>(v.d);
        if (type_ == DataType::kInt32) return v.i;
        if (type_ == DataType::kInt64) return static_cast<int32_t>(v.l);
        if (type_ == DataType::kBool) return v.b ? 1 : 0;
        THROW_INVALID_ARG("Cannot convert Scalar to int32 from type: {}", static_cast<int>(type_));
    }
    
    int64_t toInt64() const {
        if (type_ == DataType::kFloat32) return static_cast<int64_t>(v.f);
        if (type_ == DataType::kFloat64) return static_cast<int64_t>(v.d);
        if (type_ == DataType::kInt32) return static_cast<int64_t>(v.i);
        if (type_ == DataType::kInt64) return v.l;
        if (type_ == DataType::kBool) return v.b ? 1 : 0;
        THROW_INVALID_ARG("Cannot convert Scalar to int64 from type: {}", static_cast<int>(type_));
    }
    
    bool toBool() const {
        if (type_ == DataType::kFloat32) return v.f != 0.0f;
        if (type_ == DataType::kFloat64) return v.d != 0.0;
        if (type_ == DataType::kInt32) return v.i != 0;
        if (type_ == DataType::kInt64) return v.l != 0;
        if (type_ == DataType::kBool) return v.b;
        THROW_INVALID_ARG("Cannot convert Scalar to bool from type: {}", static_cast<int>(type_));
    }
    
    // 转换为data_t（保持向后兼容）
    data_t toDataT() const {
        return toFloat();  // data_t是float的别名
    }
    
    // 类型查询
    DataType dtype() const { return type_; }
    
    bool isFloatingPoint() const {
        return type_ == DataType::kFloat32 || type_ == DataType::kFloat64;
    }
    
    bool isIntegral() const {
        return type_ == DataType::kInt32 || type_ == DataType::kInt64 || type_ == DataType::kInt8;
    }
    
    bool isBool() const {
        return type_ == DataType::kBool;
    }
    
    // 算术运算符重载
    Scalar operator+(const Scalar& other) const;
    Scalar operator-(const Scalar& other) const;
    Scalar operator*(const Scalar& other) const;
    Scalar operator/(const Scalar& other) const;
    
    // 比较运算符
    bool operator==(const Scalar& other) const;
    bool operator!=(const Scalar& other) const;
    bool operator<(const Scalar& other) const;
    bool operator>(const Scalar& other) const;
    
    // 字符串表示
    std::string toString() const;
    
        /**
         * @brief 幂函数
         * @param exponent 指数
         * @return 幂运算结果
         */
        virtual std::unique_ptr<Mat> pow(data_t exponent) const = 0;
};

// 需要扩展DataType枚举以支持新类型
// 在basic_types.h中添加：
// enum class DataType {
//     kFloat32 = 0,
//     kFloat64 = 1,
//     kDouble  = 1,
//     kInt32   = 2,
//     kInt8    = 3,
//     kInt64   = 4,  // 新增
//     kBool    = 5   // 新增
// };

} 