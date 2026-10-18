// SPDX-License-Identifier: MPL-2.0
// Ported from V-Sekai-fire/service-language-model tray/src/install-portable.cpp at 2365a9d0.
// Restores payload-linux.caidx with desync from the release's shared chunk store volumes.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;

namespace fs = std::filesystem;
namespace
{
const std::string IndexName = "payload-linux.caidx";
const std::string DesyncName = "desync";
constexpr char Magic[8] = {'D', 'S', 'B', 'L', 'O', 'B', '1', '\n'};
constexpr uint64_t MaxAssetBytes = 1900000000;
constexpr size_t MaxAssetCount = 1000;

bool getText(const std::string& url, std::string& body)
{
    int pipefd[2];
    if (pipe(pipefd) != 0)
        return false;

    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0)
    {
        close(pipefd[0]);
        close(pipefd[1]);
        return false;
    }
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);

    char curl[] = "curl";
    char fail[] = "--fail";
    char location[] = "--location";
    char silent[] = "--silent";
    char showError[] = "--show-error";
    char userAgent[] = "--user-agent";
    char userAgentValue[] = "ServiceLanguageModelInstaller/1.0";
    char* const arguments[] = {curl, fail, location, silent, showError, userAgent, userAgentValue,
                               const_cast<char*>(url.c_str()), nullptr};
    pid_t child = -1;
    const int spawnError = posix_spawnp(&child, curl, &actions, nullptr, arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);
    if (spawnError != 0)
    {
        close(pipefd[0]);
        return false;
    }

    bool success = true;
    std::array<char, 1 << 16> buffer{};
    ssize_t received = 0;
    while ((received = read(pipefd[0], buffer.data(), buffer.size())) > 0)
    {
        body.append(buffer.data(), static_cast<size_t>(received));
        if (body.size() > 16 * 1024 * 1024)
        {
            success = false;
            break;
        }
    }
    if (received < 0)
        success = false;
    close(pipefd[0]);

    int status = 0;
    while (waitpid(child, &status, 0) < 0)
    {
        if (errno != EINTR)
        {
            success = false;
            break;
        }
    }
    return success && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool downloadFile(const std::string& url, const fs::path& destination)
{
    char curl[] = "curl";
    char fail[] = "--fail";
    char location[] = "--location";
    char silent[] = "--silent";
    char showError[] = "--show-error";
    char output[] = "--output";
    const std::string path = destination.string();
    char* const arguments[] = {curl, fail, location, silent, showError, output,
                               const_cast<char*>(path.c_str()), const_cast<char*>(url.c_str()), nullptr};
    pid_t child = -1;
    if (posix_spawnp(&child, curl, nullptr, nullptr, arguments, environ) != 0)
        return false;
    int status = 0;
    while (waitpid(child, &status, 0) < 0)
        if (errno != EINTR)
            return false;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        std::error_code ec;
        fs::remove(destination, ec);
        return false;
    }
    std::error_code ec;
    return fs::is_regular_file(destination, ec) && !ec && fs::file_size(destination, ec) <= MaxAssetBytes && !ec;
}

bool runDesync(const fs::path& executable, const std::vector<std::string>& arguments)
{
    const std::string executableName = executable.string();
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 2);
    argv.push_back(const_cast<char*>(executableName.c_str()));
    for (const auto& argument : arguments)
        argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    pid_t child = -1;
    if (posix_spawn(&child, executableName.c_str(), nullptr, nullptr, argv.data(), environ) != 0)
        return false;
    int status = 0;
    while (waitpid(child, &status, 0) < 0)
        if (errno != EINTR)
            return false;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool validChunkPath(const std::string& name)
{
    if (name.empty() || name.front() == '/' || name.find('\\') != std::string::npos ||
        name.find(':') != std::string::npos || name.find('\0') != std::string::npos)
        return false;
    const fs::path path(name);
    if (path.is_absolute())
        return false;
    for (const auto& part : path)
        if (part == "." || part == "..")
            return false;
    return true;
}

bool readU32(std::istream& input, uint32_t& value)
{
    std::array<unsigned char, 4> bytes{};
    if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        return false;
    value = 0;
    for (unsigned int i = 0; i < bytes.size(); ++i)
        value |= static_cast<uint32_t>(bytes[i]) << (i * 8);
    return true;
}

bool readU64(std::istream& input, uint64_t& value)
{
    std::array<unsigned char, 8> bytes{};
    if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        return false;
    value = 0;
    for (unsigned int i = 0; i < bytes.size(); ++i)
        value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    return true;
}

bool unpackVolume(const fs::path& volume, const fs::path& store)
{
    std::ifstream input(volume, std::ios::binary);
    std::array<char, sizeof(Magic)> magic{};
    uint64_t entries = 0;
    if (!input.read(magic.data(), magic.size()) || magic != std::array<char, sizeof(Magic)>{
                                                               'D', 'S', 'B', 'L', 'O', 'B', '1', '\n'} ||
        !readU64(input, entries))
        return false;

    for (uint64_t i = 0; i < entries; ++i)
    {
        uint32_t nameSize = 0;
        uint64_t fileSize = 0;
        if (!readU32(input, nameSize) || !readU64(input, fileSize) || nameSize == 0 || nameSize > 4096)
            return false;
        std::string name(nameSize, '\0');
        if (!input.read(name.data(), nameSize) || !validChunkPath(name))
            return false;

        const fs::path destination = store / fs::path(name);
        std::error_code ec;
        fs::create_directories(destination.parent_path(), ec);
        if (ec || fs::exists(destination))
            return false;
        std::ofstream output(destination, std::ios::binary | std::ios::trunc);
        if (!output)
            return false;

        std::array<char, 1 << 16> buffer{};
        uint64_t remaining = fileSize;
        while (remaining > 0)
        {
            const auto count = static_cast<std::streamsize>(std::min<uint64_t>(remaining, buffer.size()));
            if (!input.read(buffer.data(), count))
                return false;
            output.write(buffer.data(), count);
            if (!output)
                return false;
            remaining -= static_cast<uint64_t>(count);
        }
    }
    return input.peek() == std::char_traits<char>::eof();
}

bool validateVolumes(const fs::path& assets, std::vector<fs::path>& volumes)
{
    const std::regex pattern(R"(payload-data-([0-9]{3})\.bin)");
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(assets, ec))
    {
        if (ec)
            return false;
        if (entry.is_regular_file(ec) && !ec &&
            std::regex_match(entry.path().filename().string(), pattern))
            volumes.push_back(entry.path());
    }
    if (ec || volumes.empty() || volumes.size() > MaxAssetCount)
        return false;

    std::sort(volumes.begin(), volumes.end());
    for (size_t i = 0; i < volumes.size(); ++i)
    {
        char expected[32]{};
        std::snprintf(expected, sizeof(expected), "payload-data-%03zu.bin", i);
        if (volumes[i].filename() != expected)
            return false;
    }
    return true;
}

bool downloadRelease(const std::string& repository, const std::string& tag, const fs::path& assets)
{
    if (!std::regex_match(tag, std::regex(R"((v[0-9]+\.[0-9]+\.[0-9]+-)?dev\.[A-Za-z0-9.-]+)")) ||
        !std::regex_match(repository, std::regex(R"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")))
        return false;

    std::string json;
    if (!getText("https://api.github.com/repos/" + repository + "/releases/tags/" + tag, json))
        return false;

    const std::regex assetPattern(R"re("browser_download_url"\s*:\s*"([^"]+)")re");
    const std::regex volume(R"(payload-data-[0-9]{3}\.bin)");
    std::map<std::string, std::string> urls;
    for (std::sregex_iterator it(json.begin(), json.end(), assetPattern), end; it != end; ++it)
    {
        const std::string url = (*it)[1];
        const std::string prefix =
            "https://github.com/" + repository + "/releases/download/" + tag + "/";
        if (url.rfind(prefix, 0) != 0)
            continue;
        const std::string name = url.substr(prefix.size());
        if (name == DesyncName || name == IndexName || std::regex_match(name, volume))
            urls[name] = url;
    }
    if (urls.size() < 3 || !urls.contains(IndexName))
        return false;
    const std::string& desyncName = DesyncName;
    if (!urls.contains(desyncName))
        return false;

    for (const auto& [name, url] : urls)
    {
        if (!downloadFile(url, assets / name))
        {
            std::cerr << "Failed to download release asset: " << name << '\n';
            return false;
        }
    }
    std::error_code ec;
    fs::permissions(assets / desyncName, fs::perms::owner_exec | fs::perms::group_exec |
                                              fs::perms::others_exec,
                    fs::perm_options::add, ec);
    if (ec)
        return false;
    std::vector<fs::path> volumes;
    return fs::is_regular_file(assets / IndexName) &&
           fs::is_regular_file(assets / desyncName) && validateVolumes(assets, volumes);
}

int install(const fs::path& assets, const fs::path& target)
{
    const fs::path desync = assets / DesyncName;
    const fs::path index = assets / IndexName;
    std::vector<fs::path> volumes;
    if (!fs::is_regular_file(desync) || !fs::is_regular_file(index) || !validateVolumes(assets, volumes))
    {
        std::cerr << "Incomplete asset bundle; provide " << DesyncName << ", " << IndexName << ", and every "
                     "payload-data-*.bin volume.\n";
        return 1;
    }

    std::error_code ec;
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto processId = getpid();
    const fs::path store = fs::temp_directory_path() /
                           ("ninfer-all-" + std::to_string(processId) + "-" +
                            std::to_string(stamp));
    if (!fs::create_directory(store, ec) || ec)
    {
        std::cerr << "Cannot create temporary desync store: " << store << '\n';
        return 1;
    }

    bool success = true;
    for (const auto& volume : volumes)
        if (!unpackVolume(volume, store))
        {
            std::cerr << "Invalid or truncated data volume: " << volume.filename() << '\n';
            success = false;
            break;
        }

    if (success && !runDesync(desync, {"verify", "--store", store.string()}))
    {
        std::cerr << "Desync rejected the reconstructed chunk store.\n";
        success = false;
    }
    if (success)
    {
        fs::create_directories(target, ec);
        if (ec)
        {
            std::cerr << "Cannot create install directory: " << target << '\n';
            success = false;
        }
    }
    // --no-same-owner: the payload records the build host's owner (uid 1000 on the release
    // host), and an installing user without that uid cannot chown to it.
    if (success && !runDesync(desync, {"untar", "--no-same-owner", "--index", "--store", store.string(),
                                       index.string(), target.string()}))
    {
        std::cerr << "Desync could not restore the indexed payload.\n";
        success = false;
    }
    fs::remove_all(store, ec);
    // The payload is a tar index, so restore the execute bits of the shipped programs.
    if (success)
        for (const auto& entry : fs::directory_iterator(target / "bin", ec))
            if (entry.is_regular_file())
                fs::permissions(entry.path(), fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                                fs::perm_options::add, ec);
    if (success)
        std::cout << "Installation complete: " << target << '\n';
    return success ? 0 : 1;
}
} // namespace

int main(int argc, char** argv)
{
    fs::path assets;
    fs::path target = fs::current_path();
    std::string tag;
    std::string repository = "V-Sekai-fire/ninfer-all";
    bool online = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];
        if ((option == "--from" || option == "--target" || option == "--tag" || option == "--repo") &&
            i + 1 < argc)
        {
            const std::string value = argv[++i];
            if (option == "--from")
                assets = value;
            else if (option == "--target")
                target = value;
            else if (option == "--tag")
                tag = value;
            else
                repository = value;
        }
        else if (option == "--online")
            online = true;
        else
        {
            std::cerr << "Usage: ninfer-install [--from <asset-folder> | --online] "
                         "[--tag vX.Y.Z-dev.<n>] [--repo <owner/name>] [--target <folder>]\n";
            return 2;
        }
    }

    if (online && !assets.empty())
    {
        std::cerr << "Choose either --online or --from, not both.\n";
        return 2;
    }
    fs::path downloadedAssets;
    if (online || assets.empty())
    {
        if (tag.empty())
        {
            std::cerr << "Online installation requires --tag vX.Y.Z-dev.<n>.\n";
            return 2;
        }
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        downloadedAssets = fs::temp_directory_path() /
                           ("ninfer-all-download-" + std::to_string(stamp));
        std::error_code ec;
        if (!fs::create_directory(downloadedAssets, ec) || ec)
        {
            std::cerr << "Cannot create temporary download directory: " << downloadedAssets << '\n';
            return 1;
        }
        if (!downloadRelease(repository, tag, downloadedAssets))
        {
            fs::remove_all(downloadedAssets, ec);
            std::cerr << "Could not download a complete desync release bundle.\n";
            return 1;
        }
        assets = downloadedAssets;
    }

    const int result = install(fs::absolute(assets), fs::absolute(target));
    if (!downloadedAssets.empty())
    {
        std::error_code ec;
        fs::remove_all(downloadedAssets, ec);
    }
    return result;
}