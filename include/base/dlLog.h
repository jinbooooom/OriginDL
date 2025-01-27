#ifndef __ORIGINDL_LOG_H__
#define __ORIGINDL_LOG_H__

#define SPDLOG_ACTIVE_LEVEL \
    SPDLOG_LEVEL_TRACE  // 必须定义这个宏,才能输出文件名和行号，避免 spdlog 先定义该值，所以定义放在
                        // include spd

#include "spdlog/cfg/env.h"   // support for loading levels from the environment variable
#include "spdlog/fmt/ostr.h"  // support for user defined types
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"


class DLLog final
{

protected:
    DLLog() {
        static auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::trace);
        console_sink->set_pattern("%^%W %Y-%m-%d %H:%M:%S.%e %^%L %t %P [%s:%!:%#] %v%$");

        static auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("mylogs/multisink.txt", true);
        file_sink->set_level(spdlog::level::trace);
        file_sink->set_pattern("%^%W %Y-%m-%d %H:%M:%S.%e %^%L %t %P [%s:%!:%#] %v%$");

        static spdlog::logger sLogger("multi_sink", {console_sink, file_sink});
        sLogger.set_level(spdlog::level::trace);

        mLogger = &sLogger;


    };
    DLLog(const DLLog &) = delete;
    DLLog &operator=(const DLLog &) = delete;
    virtual ~DLLog(){};

public:
    static DLLog *GetInstance()
    {
        static DLLog instance;
        return &instance;
    }

    // std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> console_sink;
    // std::shared_ptr<spdlog::sinks::basic_file_sink_mt> file_sink;
    spdlog::logger *mLogger;
    
};

#define logger (DLLog::GetInstance()->mLogger)
#define logt(format, ...) SPDLOG_LOGGER_TRACE(logger, format, ##__VA_ARGS__)
#define logd(format, ...) SPDLOG_LOGGER_DEBUG(logger, format, ##__VA_ARGS__)
#define logi(format, ...) SPDLOG_LOGGER_INFO(logger, format, ##__VA_ARGS__)
#define logw(format, ...) SPDLOG_LOGGER_WARN(logger, format, ##__VA_ARGS__)
#define loge(format, ...) SPDLOG_LOGGER_ERROR(logger, format, ##__VA_ARGS__)
#define logc(format, ...) SPDLOG_LOGGER_CRITICAL(logger, format, ##__VA_ARGS__)

#endif