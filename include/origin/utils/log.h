#ifndef __ORIGINDL_LOG_H__
#define __ORIGINDL_LOG_H__

#define SPDLOG_ACTIVE_LEVEL \
    SPDLOG_LEVEL_TRACE  // 必须定义这个宏,才能输出文件名和行号，避免 spdlog 先定义该值，所以定义放在
                        // include spd

#ifndef SPDLOG_COMPILED_LIB
#    define SPDLOG_COMPILED_LIB  // 使用预编译库，避免重复编译模板
#endif

#include "spdlog/cfg/env.h"   // support for loading levels from the environment variable
#include "spdlog/fmt/ostr.h"  // support for user defined types

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "spdlog/details/os.h"
#include <algorithm>
#include <cstring>

class OriginLog final
{
protected:
    /**
     * @brief 从环境变量 ORIGIN_LOG_LEVEL 读取日志级别，默认返回 WARN
     * @details 支持的环境变量值（不区分大小写）：
     *   - "trace"   : 显示所有日志（TRACE、DEBUG、INFO、WARN、ERROR、CRITICAL）
     *   - "debug"   : 显示 DEBUG 及以上的日志（DEBUG、INFO、WARN、ERROR、CRITICAL）
     *   - "info"    : 显示 INFO 及以上的日志（INFO、WARN、ERROR、CRITICAL）
     *   - "warn"    : 显示 WARN 及以上的日志（WARN、ERROR、CRITICAL）- 默认值
     *   - "error"   : 只显示 ERROR 和 CRITICAL 日志
     *   - "critical": 只显示 CRITICAL 日志
     *   - "off"     : 关闭所有日志输出
     * @return 解析后的日志级别，如果环境变量未设置或无效，返回 WARN
     * @example
     *   export ORIGIN_LOG_LEVEL=trace   # 显示所有日志
     *   export ORIGIN_LOG_LEVEL=info     # 只显示 INFO 及以上
     *   export ORIGIN_LOG_LEVEL=warn     # 只显示 WARN 及以上（默认）
     */
    static spdlog::level::level_enum get_log_level_from_env()
    {
        auto env_val = spdlog::details::os::getenv("ORIGIN_LOG_LEVEL");
        if (!env_val.empty())
        {
            std::string level_str = env_val;
            std::transform(level_str.begin(), level_str.end(), level_str.begin(), ::tolower);
            auto level = spdlog::level::from_str(level_str);
            // 如果解析成功（不是 off 或无效值），返回该级别
            if (level != spdlog::level::off || level_str == "off")
            {
                return level;
            }
        }
        // 默认返回 WARN 级别
        return spdlog::level::warn;
    }

    OriginLog()
    {
        // 从环境变量获取日志级别，默认为 WARN
        auto log_level = get_log_level_from_env();

        static auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(log_level);
        console_sink->set_pattern("%^%W %Y-%m-%d %H:%M:%S.%e %^%L %t %P [%s:%!:%#] %v%$");

        static auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("originlog/origin.txt", true);
        file_sink->set_level(log_level);
        file_sink->set_pattern("%^%W %Y-%m-%d %H:%M:%S.%e %^%L %t %P [%s:%!:%#] %v%$");

        static spdlog::logger sLogger("multi_sink", {console_sink, file_sink});
        sLogger.set_level(log_level);

        mLogger = &sLogger;
    };
    OriginLog(const OriginLog &)            = delete;
    OriginLog &operator=(const OriginLog &) = delete;
    virtual ~OriginLog(){};

public:
    static OriginLog *GetInstance()
    {
        static OriginLog instance;
        return &instance;
    }

    // std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> console_sink;
    // std::shared_ptr<spdlog::sinks::basic_file_sink_mt> file_sink;
    spdlog::logger *mLogger;
};

#define origin_logger (OriginLog::GetInstance()->mLogger)
#define logt(format, ...) SPDLOG_LOGGER_TRACE(origin_logger, format, ##__VA_ARGS__)
#define logd(format, ...) SPDLOG_LOGGER_DEBUG(origin_logger, format, ##__VA_ARGS__)
#define logi(format, ...) SPDLOG_LOGGER_INFO(origin_logger, format, ##__VA_ARGS__)
#define logw(format, ...) SPDLOG_LOGGER_WARN(origin_logger, format, ##__VA_ARGS__)
#define loge(format, ...) SPDLOG_LOGGER_ERROR(origin_logger, format, ##__VA_ARGS__)
#define logc(format, ...) SPDLOG_LOGGER_CRITICAL(origin_logger, format, ##__VA_ARGS__)

#endif
