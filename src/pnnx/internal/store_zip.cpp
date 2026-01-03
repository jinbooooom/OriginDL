// StoreZipReader 实现
#include "origin/pnnx/internal/store_zip.hpp"
#include <cstdio>
#include <cstring>
#include <cstdint>

namespace origin
{
namespace pnnx
{
namespace internal
{

StoreZipReader::StoreZipReader() : fp(nullptr) {}

StoreZipReader::~StoreZipReader()
{
    close();
}

int StoreZipReader::open(const std::string &path)
{
    close();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        return -1;
    }

    while (!feof(fp))
    {
        // peek signature
        uint32_t signature;
        int nread = fread((char *)&signature, sizeof(signature), 1, fp);
        if (nread != 1)
            break;

        if (signature == 0x04034b50)
        {
            local_file_header lfh;
            fread((char *)&lfh, sizeof(lfh), 1, fp);

            if (lfh.flag & 0x08)
            {
                // zip file contains data descriptor, this is not supported yet
                return -1;
            }

            if (lfh.compression != 0 || lfh.compressed_size != lfh.uncompressed_size)
            {
                // not stored zip file
                return -1;
            }

            // file name
            std::string name;
            name.resize(lfh.file_name_length);
            fread((char *)name.data(), name.size(), 1, fp);

            // skip extra field
            fseek(fp, lfh.extra_field_length, SEEK_CUR);

            StoreZipMeta fm;
            fm.offset = ftell(fp);
            fm.size = lfh.compressed_size;

            filemetas[name] = fm;

            fseek(fp, lfh.compressed_size, SEEK_CUR);
        }
        else if (signature == 0x02014b50)
        {
            central_directory_file_header cdfh;
            fread((char *)&cdfh, sizeof(cdfh), 1, fp);

            // skip file name
            fseek(fp, cdfh.file_name_length, SEEK_CUR);

            // skip extra field
            fseek(fp, cdfh.extra_field_length, SEEK_CUR);

            // skip file comment
            fseek(fp, cdfh.file_comment_length, SEEK_CUR);
        }
        else if (signature == 0x06054b50)
        {
            end_of_central_directory_record eocdr;
            fread((char *)&eocdr, sizeof(eocdr), 1, fp);

            // skip comment
            fseek(fp, eocdr.comment_length, SEEK_CUR);
        }
        else
        {
            // unsupported signature
            return -1;
        }
    }

    return 0;
}

size_t StoreZipReader::get_file_size(const std::string &name)
{
    if (filemetas.find(name) == filemetas.end())
    {
        return 0;
    }

    return filemetas[name].size;
}

int StoreZipReader::read_file(const std::string &name, char *data)
{
    if (filemetas.find(name) == filemetas.end())
    {
        return -1;
    }

    size_t offset = filemetas[name].offset;
    size_t size = filemetas[name].size;

    fseek(fp, offset, SEEK_SET);
    fread(data, size, 1, fp);

    return 0;
}

int StoreZipReader::close()
{
    if (fp)
    {
        fclose(fp);
        fp = nullptr;
    }
    filemetas.clear();
    return 0;
}

}  // namespace internal
}  // namespace pnnx
}  // namespace origin
