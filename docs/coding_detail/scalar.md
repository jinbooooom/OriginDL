// 适合OriginDL的简化Scalar实现
namespace origin {

class Scalar {
private:
    union {
        float f;       // 单精度浮点数
        double d;      // 双精度浮点数
        int32_t i;     // 32位整数
        int64_t l;     // 64位整数
        bool b;        // 布尔值
    } v;
    DataType type_;    // 使用OriginDL现有的DataType
    
public:
    // 构造函数
    Scalar(float f) : v{f}, type_(DataType::kFloat32) {}
    Scalar(double d) : v{d}, type_(DataType::kFloat64) {}
    Scalar(int32_t i) : v{i}, type_(DataType::kInt32) {}
    Scalar(int64_t l) : v{l}, type_(DataType::kInt64) {}  // 需要添加kInt64
    Scalar(bool b) : v{b}, type_(DataType::kBool) {}      // 需要添加kBool
    
    // 类型转换
    float toFloat() const;
    double toDouble() const;
    int32_t toInt32() const;
    int64_t toInt64() const;
    bool toBool() const;
    
    // 转换为data_t（保持向后兼容）
    data_t toDataT() const;
    
    // 类型查询
    DataType dtype() const { return type_; }
    bool isFloatingPoint() const;
    bool isIntegral() const;
};

} 