#include "originDL.h"
#include "base/dlException.h"
#include <iostream>
#include <stdexcept>

using namespace dl;

void testWarningException()
{
    std::cout << "=== Testing Warning Exception ===" << std::endl;
    try
    {
        DL_WARN_THROW("This is a warning exception test");
    }
    catch (const DLWarningException& e)
    {
        std::cout << "Caught DLWarningException: " << e.what() << std::endl;
        std::cout << "File: " << e.file() << ", Line: " << e.line() << ", Function: " << e.function() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught std::exception: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

void testErrorException()
{
    std::cout << "=== Testing Error Exception ===" << std::endl;
    try
    {
        DL_ERROR_THROW("This is an error exception test");
    }
    catch (const DLErrorException& e)
    {
        std::cout << "Caught DLErrorException: " << e.what() << std::endl;
        std::cout << "File: " << e.file() << ", Line: " << e.line() << ", Function: " << e.function() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught std::exception: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

void testCriticalException()
{
    std::cout << "=== Testing Critical Exception ===" << std::endl;
    try
    {
        DL_CRITICAL_THROW("This is a critical exception test");
    }
    catch (const DLCriticalException& e)
    {
        std::cout << "Caught DLCriticalException: " << e.what() << std::endl;
        std::cout << "File: " << e.file() << ", Line: " << e.line() << ", Function: " << e.function() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught std::exception: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

void testTensorException()
{
    std::cout << "=== Testing Tensor Exception (Simulated) ===" << std::endl;
    try
    {
        // 模拟一个会导致异常的张量操作
        auto x = std::make_shared<Variable>(af::constant(1.0, af::dim4(2, 2)));
        
        // 故意创建一个无效的梯度来触发异常
        // 这里我们模拟一个会导致 critical 异常的情况
        DL_CRITICAL_THROW("Simulated tensor operation failure");
    }
    catch (const DLCriticalException& e)
    {
        std::cout << "Caught DLCriticalException in tensor operation: " << e.what() << std::endl;
        std::cout << "File: " << e.file() << ", Line: " << e.line() << ", Function: " << e.function() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught std::exception in tensor operation: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

void testOperatorException()
{
    std::cout << "=== Testing Operator Exception (Simulated) ===" << std::endl;
    try
    {
        // 模拟一个会导致异常的操作符
        auto x = std::make_shared<Variable>(af::constant(1.0, af::dim4(2, 2)));
        auto y = std::make_shared<Variable>(af::constant(2.0, af::dim4(2, 2)));
        
        // 故意触发一个警告异常
        DL_WARN_THROW("Simulated operator parameter validation failure");
    }
    catch (const DLWarningException& e)
    {
        std::cout << "Caught DLWarningException in operator: " << e.what() << std::endl;
        std::cout << "File: " << e.file() << ", Line: " << e.line() << ", Function: " << e.function() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught std::exception in operator: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

void testExceptionHierarchy()
{
    std::cout << "=== Testing Exception Hierarchy ===" << std::endl;
    
    // 测试异常继承关系
    try
    {
        DL_ERROR_THROW("Testing exception hierarchy");
    }
    catch (const DLException& e)
    {
        std::cout << "Caught base DLException: " << e.what() << std::endl;
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "Caught std::runtime_error: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught std::exception: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "=== DL Exception Handling Test ===" << std::endl;
    std::cout << "Testing exception handling system with logging integration" << std::endl;
    std::cout << std::endl;
    
    // 设置后端
    af::setBackend(AF_BACKEND_CPU);
    
    // 运行各种异常测试
    testWarningException();
    testErrorException();
    testCriticalException();
    testTensorException();
    testOperatorException();
    testExceptionHierarchy();
    
    std::cout << "=== All Exception Tests Completed ===" << std::endl;
    std::cout << "Check the log file 'dllog/dl.txt' for detailed logging information" << std::endl;
    
    return 0;
}
