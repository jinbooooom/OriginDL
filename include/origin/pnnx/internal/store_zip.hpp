// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef __ORIGIN_DL_PNNX_STOREZIP_H__
#define __ORIGIN_DL_PNNX_STOREZIP_H__

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace origin
{
namespace pnnx
{
namespace internal
{

// ZIP 文件格式结构体定义
#ifdef _MSC_VER
#    define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#else
#    define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

PACK(struct local_file_header {
    uint16_t version;
    uint16_t flag;
    uint16_t compression;
    uint16_t last_modify_time;
    uint16_t last_modify_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t file_name_length;
    uint16_t extra_field_length;
});

PACK(struct central_directory_file_header {
    uint16_t version_made;
    uint16_t version;
    uint16_t flag;
    uint16_t compression;
    uint16_t last_modify_time;
    uint16_t last_modify_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t file_name_length;
    uint16_t extra_field_length;
    uint16_t file_comment_length;
    uint16_t start_disk;
    uint16_t internal_file_attrs;
    uint32_t external_file_attrs;
    uint32_t lfh_offset;
});

PACK(struct end_of_central_directory_record {
    uint16_t disk_number;
    uint16_t start_disk;
    uint16_t cd_records;
    uint16_t total_cd_records;
    uint32_t cd_size;
    uint32_t cd_offset;
    uint16_t comment_length;
});

class StoreZipReader
{
public:
    StoreZipReader();
    ~StoreZipReader();

    int open(const std::string &path);

    size_t get_file_size(const std::string &name);

    int read_file(const std::string &name, char *data);

    int close();

private:
    FILE *fp;

    struct StoreZipMeta
    {
        size_t offset;
        size_t size;
    };

    std::map<std::string, StoreZipMeta> filemetas;
};

class StoreZipWriter
{
public:
    StoreZipWriter();
    ~StoreZipWriter();

    int open(const std::string &path);

    int write_file(const std::string &name, const char *data, size_t size);

    int close();

private:
    FILE *fp;

    struct StoreZipMeta
    {
        std::string name;
        size_t lfh_offset;
        uint32_t crc32;
        uint32_t size;
    };

    std::vector<StoreZipMeta> filemetas;
};

}  // namespace internal
}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_STOREZIP_H__
