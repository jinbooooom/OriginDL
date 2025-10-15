#include "origin/utils/exception.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "origin/utils/log.h"

using namespace origin;

// 模拟一些工具函数
std::string dtype_to_string(int dtype)
{
    switch (dtype)
    {
        case 0:
            return "float32";
        case 1:
            return "int32";
        case 2:
            return "int8";
        default:
            return "unknown";
    }
}

std::string format_shape(const std::vector<int> &shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

void testBasicThrow()
{
    logw("=== Testing Basic THROW ===");
    try
    {
        THROW(std::invalid_argument, "Basic exception test");
    }
    catch (const std::invalid_argument &e)
    {
        logi("Caught std::invalid_argument: {}", e.what());
    }
}

void testFormattedThrow()
{
    logw("=== Testing Formatted THROW ===");
    try
    {
        int dtype = 0;
        THROW(std::invalid_argument, "not support dtype: {}", dtype_to_string(dtype));
    }
    catch (const std::invalid_argument &e)
    {
        logi("Caught std::invalid_argument: {}", e.what());
    }
}

void testMultipleArguments()
{
    logw("=== Testing Multiple Arguments ===");
    try
    {
        std::vector<int> expected_shape = {2, 3};
        std::vector<int> actual_shape   = {3, 2};
        THROW(std::invalid_argument, "Expected shape {} but got {}", format_shape(expected_shape),
              format_shape(actual_shape));
    }
    catch (const std::invalid_argument &e)
    {
        logi("Caught std::invalid_argument: {}", e.what());
    }
}

void testSimplifiedMacros()
{
    logw("=== Testing Simplified Macros ===");

    // 测试 THROW_INVALID_ARG
    try
    {
        THROW_INVALID_ARG("Data type mismatch for add operation");
    }
    catch (const std::invalid_argument &e)
    {
        logi("Caught THROW_INVALID_ARG: {}", e.what());
    }

    // 测试 THROW_RUNTIME_ERROR
    try
    {
        THROW_RUNTIME_ERROR("CUDA device transfer not supported yet");
    }
    catch (const std::runtime_error &e)
    {
        logi("Caught THROW_RUNTIME_ERROR: {}", e.what());
    }

    // 测试 THROW_LOGIC_ERROR
    try
    {
        THROW_LOGIC_ERROR("Invalid tensor operation");
    }
    catch (const std::logic_error &e)
    {
        logi("Caught THROW_LOGIC_ERROR: {}", e.what());
    }
}

void testConditionalThrow()
{
    logw("=== Testing Conditional THROW ===");

    try
    {
        int value = 0;
        THROW_IF(value == 0, std::invalid_argument, "Value({}) should not be zero", value);
    }
    catch (const std::invalid_argument &e)
    {
        logi("Caught THROW_IF for invalid argument: {}", e.what());
    }

    try
    {
        bool success = false;
        THROW_IF(!success, std::runtime_error, "Operation failed");
    }
    catch (const std::runtime_error &e)
    {
        logi("Caught THROW_IF for runtime error: {}", e.what());
    }

    // 测试条件为false时不抛出
    try
    {
        int value = 5;
        THROW_IF(value <= 0, std::invalid_argument, "Value should be positive");
        logi("THROW_IF with true condition: No exception thrown (correct)");
    }
    catch (const std::invalid_argument &e)
    {
        logi("Unexpected exception: {}", e.what());
    }
}

void testComplexFormattedMessages()
{
    logw("=== Testing Complex Formatted Messages ===");

    try
    {
        std::string operation = "matrix_multiply";
        std::string device    = "CUDA";
        int dtype             = 1;
        THROW(std::invalid_argument, "Unsupported operation: {} on device {} with dtype {}", operation, device,
              dtype_to_string(dtype));
    }
    catch (const std::invalid_argument &e)
    {
        logi("Caught complex formatted message: {}", e.what());
    }

    try
    {
        std::vector<int> shape_a = {2, 3};
        std::vector<int> shape_b = {3, 2};
        THROW(std::invalid_argument, "Dimension mismatch: tensor A{} vs tensor B{}", format_shape(shape_a),
              format_shape(shape_b));
    }
    catch (const std::invalid_argument &e)
    {
        logi("Caught dimension mismatch: {}", e.what());
    }

    try
    {
        double value   = 3.14;
        double min_val = 0.0;
        double max_val = 1.0;
        THROW(std::invalid_argument, "Value {} out of range [{}, {}]", value, min_val, max_val);
    }
    catch (const std::invalid_argument &e)
    {
        logi("Caught value range error: {}", e.what());
    }
}

void testDifferentExceptionTypes()
{
    logw("=== Testing Different Exception Types ===");

    // 测试 std::out_of_range
    try
    {
        THROW(std::out_of_range, "Index {} out of range", 10);
    }
    catch (const std::out_of_range &e)
    {
        logi("Caught std::out_of_range: {}", e.what());
    }

    // 测试 std::bad_alloc (注意：std::bad_alloc 不接受字符串参数)
    try
    {
        throw std::bad_alloc();
    }
    catch (const std::bad_alloc &e)
    {
        logi("Caught std::bad_alloc: {}", e.what());
    }
}

int main(int argc, char **argv)
{
    logi("Testing the redesigned exception system with formatting support");

    // 运行各种异常测试
    testBasicThrow();
    testFormattedThrow();
    testMultipleArguments();
    testSimplifiedMacros();
    testConditionalThrow();
    testComplexFormattedMessages();
    testDifferentExceptionTypes();

    logw("=== All Exception Tests Completed ===");

    return 0;
}