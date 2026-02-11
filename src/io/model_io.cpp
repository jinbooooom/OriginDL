#include "origin/io/model_io.h"
#include <fstream>
#include <sstream>
#include <vector>
#include "origin/core/tensor.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{

// 内部辅助函数：将字符串写入二进制文件
static void write_string(std::ofstream &out, const std::string &str)
{
    size_t len = str.size();
    out.write(reinterpret_cast<const char *>(&len), sizeof(len));
    out.write(str.c_str(), len);
}

// 内部辅助函数：从二进制文件读取字符串
static std::string read_string(std::ifstream &in)
{
    size_t len;
    in.read(reinterpret_cast<char *>(&len), sizeof(len));
    if (unlikely(in.fail()))
    {
        THROW_RUNTIME_ERROR("Failed to read string length from file");
    }
    std::string str(len, '\0');
    in.read(&str[0], len);
    if (unlikely(in.fail()))
    {
        THROW_RUNTIME_ERROR("Failed to read string data from file");
    }
    return str;
}

// 内部辅助函数：将 Shape 写入二进制文件
static void write_shape(std::ofstream &out, const Shape &shape)
{
    size_t dims = shape.size();
    out.write(reinterpret_cast<const char *>(&dims), sizeof(dims));
    for (size_t i = 0; i < dims; ++i)
    {
        size_t dim = shape[i];
        out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    }
}

// 内部辅助函数：从二进制文件读取 Shape
static Shape read_shape(std::ifstream &in)
{
    size_t dims;
    in.read(reinterpret_cast<char *>(&dims), sizeof(dims));
    if (in.fail())
    {
        THROW_RUNTIME_ERROR("Failed to read shape dimensions from file");
    }
    std::vector<size_t> dims_vec(dims);
    for (size_t i = 0; i < dims; ++i)
    {
        size_t dim;
        in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
        if (in.fail())
        {
            THROW_RUNTIME_ERROR("Failed to read shape dimension {} from file", i);
        }
        dims_vec[i] = dim;
    }
    return Shape(dims_vec);
}

// 内部辅助函数：将 DataType 写入二进制文件
static void write_dtype(std::ofstream &out, DataType dtype)
{
    int32_t dtype_int = static_cast<int32_t>(dtype);
    out.write(reinterpret_cast<const char *>(&dtype_int), sizeof(dtype_int));
}

// 内部辅助函数：从二进制文件读取 DataType
static DataType read_dtype(std::ifstream &in)
{
    int32_t dtype_int;
    in.read(reinterpret_cast<char *>(&dtype_int), sizeof(dtype_int));
    if (in.fail())
    {
        THROW_RUNTIME_ERROR("Failed to read dtype from file");
    }
    return static_cast<DataType>(dtype_int);
}

// 内部辅助函数：将 Device 写入二进制文件
static void write_device(std::ofstream &out, const Device &device)
{
    int32_t device_type = static_cast<int32_t>(device.type());
    out.write(reinterpret_cast<const char *>(&device_type), sizeof(device_type));
    int32_t device_index = device.index();
    out.write(reinterpret_cast<const char *>(&device_index), sizeof(device_index));
}

// 内部辅助函数：从二进制文件读取 Device
static Device read_device(std::ifstream &in)
{
    int32_t device_type_int, device_index;
    in.read(reinterpret_cast<char *>(&device_type_int), sizeof(device_type_int));
    in.read(reinterpret_cast<char *>(&device_index), sizeof(device_index));
    if (in.fail())
    {
        THROW_RUNTIME_ERROR("Failed to read device from file");
    }
    return Device(static_cast<DeviceType>(device_type_int), device_index);
}

void save(const StateDict &state_dict, const std::string &filepath)
{
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open())
    {
        THROW_RUNTIME_ERROR("Failed to open file for writing: {}", filepath);
    }

    // 写入魔数（用于格式识别）
    const uint32_t magic = 0x4F444C00;  // "ODL\0"
    out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

    // 写入格式版本
    const uint32_t version = 1;
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));

    // 写入参数数量
    size_t param_count = state_dict.size();
    out.write(reinterpret_cast<const char *>(&param_count), sizeof(param_count));

    // 写入每个参数
    for (const auto &[name, tensor] : state_dict)
    {
        // 写入参数名称
        write_string(out, name);

        // 写入张量元数据
        write_shape(out, tensor.shape());
        write_dtype(out, tensor.dtype());
        write_device(out, tensor.device());

        // 写入张量数据
        // 目前只支持 float32，其他类型需要扩展
        if (tensor.dtype() == DataType::kFloat32)
        {
            auto data        = tensor.to_vector<float>();
            size_t data_size = data.size() * sizeof(float);
            out.write(reinterpret_cast<const char *>(data.data()), data_size);
        }
        else if (tensor.dtype() == DataType::kFloat64)
        {
            auto data        = tensor.to_vector<double>();
            size_t data_size = data.size() * sizeof(double);
            out.write(reinterpret_cast<const char *>(data.data()), data_size);
        }
        else if (tensor.dtype() == DataType::kInt32)
        {
            auto data        = tensor.to_vector<int32_t>();
            size_t data_size = data.size() * sizeof(int32_t);
            out.write(reinterpret_cast<const char *>(data.data()), data_size);
        }
        else
        {
            THROW_RUNTIME_ERROR("Unsupported dtype for saving: {}", static_cast<int>(tensor.dtype()));
        }
    }

    out.close();
    if (out.fail())
    {
        THROW_RUNTIME_ERROR("Failed to write to file: {}", filepath);
    }
}

StateDict load(const std::string &filepath)
{
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open())
    {
        THROW_RUNTIME_ERROR("Failed to open file for reading: {}", filepath);
    }

    // 读取魔数
    uint32_t magic;
    in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (in.fail() || magic != 0x4F444C00)
    {
        THROW_RUNTIME_ERROR("Invalid file format or corrupted file: {}", filepath);
    }

    // 读取格式版本
    uint32_t version;
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (in.fail() || version != 1)
    {
        THROW_RUNTIME_ERROR("Unsupported file version: {}", version);
    }

    // 读取参数数量
    size_t param_count;
    in.read(reinterpret_cast<char *>(&param_count), sizeof(param_count));
    if (in.fail())
    {
        THROW_RUNTIME_ERROR("Failed to read parameter count from file");
    }

    StateDict state_dict;

    // 读取每个参数
    for (size_t i = 0; i < param_count; ++i)
    {
        // 读取参数名称
        std::string name = read_string(in);

        // 读取张量元数据
        Shape shape    = read_shape(in);
        DataType dtype = read_dtype(in);
        Device device  = read_device(in);

        // 读取张量数据
        Tensor tensor;
        if (dtype == DataType::kFloat32)
        {
            size_t element_count = shape.elements();
            std::vector<float> data(element_count);
            size_t data_size = element_count * sizeof(float);
            in.read(reinterpret_cast<char *>(data.data()), data_size);
            if (in.fail())
            {
                THROW_RUNTIME_ERROR("Failed to read tensor data for parameter '{}'", name);
            }
            tensor = Tensor(data, shape, TensorOptions().dtype(dtype).device(device));
        }
        else if (dtype == DataType::kFloat64)
        {
            size_t element_count = shape.elements();
            std::vector<double> data(element_count);
            size_t data_size = element_count * sizeof(double);
            in.read(reinterpret_cast<char *>(data.data()), data_size);
            if (in.fail())
            {
                THROW_RUNTIME_ERROR("Failed to read tensor data for parameter '{}'", name);
            }
            tensor = Tensor(data, shape, TensorOptions().dtype(dtype).device(device));
        }
        else if (dtype == DataType::kInt32)
        {
            size_t element_count = shape.elements();
            std::vector<int32_t> data(element_count);
            size_t data_size = element_count * sizeof(int32_t);
            in.read(reinterpret_cast<char *>(data.data()), data_size);
            if (in.fail())
            {
                THROW_RUNTIME_ERROR("Failed to read tensor data for parameter '{}'", name);
            }
            tensor = Tensor(data, shape, TensorOptions().dtype(dtype).device(device));
        }
        else
        {
            THROW_RUNTIME_ERROR("Unsupported dtype for loading: {}", static_cast<int>(dtype));
        }

        state_dict[name] = tensor;
    }

    in.close();
    return state_dict;
}

}  // namespace origin
