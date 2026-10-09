// SPDX-License-Identifier: MPL-2.0

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace
{
constexpr uint64_t VolumeTarget = 1'800'000'000;
constexpr uint64_t VolumeLimit = 1'900'000'000;
constexpr char Magic[8] = {'D', 'S', 'B', 'L', 'O', 'B', '1', '\n'};

void writeU32(std::ostream& out, uint32_t value)
{
    for (unsigned int i = 0; i < 4; ++i)
        out.put(static_cast<char>((value >> (i * 8)) & 0xff));
}

void writeU64(std::ostream& out, uint64_t value)
{
    for (unsigned int i = 0; i < 8; ++i)
        out.put(static_cast<char>((value >> (i * 8)) & 0xff));
}

void pack(const fs::path& store, const fs::path& output)
{
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(store))
        if (entry.is_regular_file())
            files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("The desync store contains no chunks.");

    fs::create_directories(output);
    std::ofstream volume;
    uint64_t volumeBytes = 0;
    uint64_t entries = 0;
    unsigned int volumeNumber = 0;
    std::string currentVolumeName;
    auto finishVolume = [&]() {
        if (!volume.is_open())
            return;
        const auto end = volume.tellp();
        if (end < 0 || static_cast<uint64_t>(end) >= VolumeLimit)
            throw std::runtime_error("A data volume reached the asset size limit.");
        volume.seekp(8);
        writeU64(volume, entries);
        volume.close();
        std::cout << currentVolumeName << " " << end << " bytes\n";
    };
    auto startVolume = [&]() {
        if (volumeNumber >= 1000)
            throw std::runtime_error("The chunk store needs more than 1000 release volumes.");
        const std::string number = std::to_string(volumeNumber++);
        currentVolumeName = "payload-data-" + std::string(3 - number.size(), '0') + number + ".bin";
        const fs::path path = output / currentVolumeName;
        volume.open(path, std::ios::binary | std::ios::trunc);
        if (!volume)
            throw std::runtime_error("Cannot create data volume: " + path.string());
        volume.write(Magic, sizeof(Magic));
        writeU64(volume, 0);
        volumeBytes = 16;
        entries = 0;
    };

    for (const auto& file : files)
    {
        const std::string name = fs::relative(file, store).generic_string();
        if (name.empty() || name.size() > 4096 ||
            name.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/.-_") !=
                std::string::npos)
            throw std::runtime_error("Unexpected non-ASCII chunk path: " + name);
        const uint64_t size = fs::file_size(file);
        const uint64_t entryBytes = 12 + name.size() + size;
        if (entryBytes + 16 > VolumeTarget)
            throw std::runtime_error("A desync chunk is too large for a data volume: " + name);
        if (!volume.is_open() || volumeBytes + entryBytes > VolumeTarget)
        {
            finishVolume();
            startVolume();
        }

        writeU32(volume, static_cast<uint32_t>(name.size()));
        writeU64(volume, size);
        volume.write(name.data(), static_cast<std::streamsize>(name.size()));
        std::ifstream input(file, std::ios::binary);
        if (!input)
            throw std::runtime_error("Cannot read chunk: " + file.string());
        volume << input.rdbuf();
        if (!volume || input.bad())
            throw std::runtime_error("Failed while writing chunk: " + file.string());
        volumeBytes += entryBytes;
        ++entries;
    }
    finishVolume();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: payload-pack <desync-store> <output-directory>\n";
        return 2;
    }
    try
    {
        pack(fs::absolute(argv[1]), fs::absolute(argv[2]));
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "payload-pack: " << error.what() << '\n';
        return 1;
    }
}
