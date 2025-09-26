#ifndef __ORIGINDL_EXCEPTION_H__
#define __ORIGINDL_EXCEPTION_H__

#include <stdexcept>
#include <string>
#include "dlLog.h"

namespace dl
{

// 基础异常类
class DLException : public std::runtime_error
{
public:
    DLException(const std::string &message, const char *file, int line, const char *function)
        : std::runtime_error(message), mFile(file), mLine(line), mFunction(function)
    {}

    const char *file() const { return mFile; }
    int line() const { return mLine; }
    const char *function() const { return mFunction; }

private:
    const char *mFile;
    int mLine;
    const char *mFunction;
};

// 警告级别异常（非致命）
class DLWarningException : public DLException
{
public:
    DLWarningException(const std::string &message, const char *file, int line, const char *function)
        : DLException(message, file, line, function)
    {}
};

// 错误级别异常（致命）
class DLErrorException : public DLException
{
public:
    DLErrorException(const std::string &message, const char *file, int line, const char *function)
        : DLException(message, file, line, function)
    {}
};

// 严重错误异常（系统级）
class DLCriticalException : public DLException
{
public:
    DLCriticalException(const std::string &message, const char *file, int line, const char *function)
        : DLException(message, file, line, function)
    {}
};

}  // namespace dl

// 简化的异常抛出宏：先记录日志，再抛出异常
#define DL_WARN_THROW(message)                                                   \
    do                                                                           \
    {                                                                            \
        logw("{}", message);                                                     \
        throw dl::DLWarningException(message, __FILE__, __LINE__, __FUNCTION__); \
    } while (0)

#define DL_ERROR_THROW(message)                                                \
    do                                                                         \
    {                                                                          \
        loge("{}", message);                                                   \
        throw dl::DLErrorException(message, __FILE__, __LINE__, __FUNCTION__); \
    } while (0)

#define DL_CRITICAL_THROW(message)                                                \
    do                                                                            \
    {                                                                             \
        logc("{}", __FUNCTION__, message);                                        \
        throw dl::DLCriticalException(message, __FILE__, __LINE__, __FUNCTION__); \
    } while (0)

#endif
