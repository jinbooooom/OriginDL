#include "originDL.h"

using namespace dl;

int main()
{
    int value  = 99;
    size_t size = 100;
    void *p     = &size;

    logt("addr = {}, size = {}", p, size);
    logd("addr = {}, size = {}", p, size);
    logi("addr = {}, size = {}", p, size);
    logw("addr = {}, size = {}", p, size);
    loge("addr = {}, size = {}", p, size);
    logc("addr = {}, size = {}", p, size);

    logd("");
    logd("Easy padding in numbers like {:08d}", value);
    logd("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
    logd("Support for floats {:03.2f}", 1.23456);
    logd("Positional args are {1} {0}..", "too", "supported");
    logd("{:>8} aligned, {:<8} aligned", "right", "left");

    return 0;
}
