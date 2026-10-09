// SPDX-License-Identifier: MPL-2.0
// Ported from V-Sekai-fire/service-language-model tray/src/install-portable.cpp at 2365a9d0.
// One release carries both platforms over one shared chunk store: Windows restores
// payload.caidx with desync.exe, Linux restores payload-linux.caidx with desync.

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

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#else
#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace fs = std::filesystem;
namespace
{
#ifdef _WIN32
const std::string IndexName = "payload.caidx";
const std::string DesyncName = "desync.exe";
#else
const std::string IndexName = "payload-linux.caidx";
const std::string DesyncName = "desync";
#endif
constexpr char Magic[8] = {'D', 'S', 'B', 'L', 'O', 'B', '1', '\n'};
constexpr uint64_t MaxAssetBytes = 1900000000;
constexpr size_t MaxAssetCount = 1000;

#ifdef _WIN32
using InternetHandle = std::unique_ptr<void, decltype(&WinHttpCloseHandle)>;

std::wstring widen(const std::string& value)
{
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(),
                                         static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0)
        return {};
    std::wstring result(static_cast<size_t>(size), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(),
                            static_cast<int>(value.size()), result.data(), size) != size)
        return {};
    return result;
}

InternetHandle openRequest(HINTERNET session, const std::wstring& url)
{
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof(parts);
    parts.dwSchemeLength = static_cast<DWORD>(-1);
    parts.dwHostNameLength = static_cast<DWORD>(-1);
    parts.dwUrlPathLength = static_cast<DWORD>(-1);
    parts.dwExtraInfoLength = static_cast<DWORD>(-1);
    if (!WinHttpCrackUrl(url.c_str(), 0, 0, &parts))
        return {nullptr, WinHttpCloseHandle};

    const std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
    path.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);
    InternetHandle connection{WinHttpConnect(session, host.c_str(), parts.nPort, 0), WinHttpCloseHandle};
    if (!connection)
        return {nullptr, WinHttpCloseHandle};
    return {WinHttpOpenRequest(connection.get(), L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
                               WINHTTP_DEFAULT_ACCEPT_TYPES,
                               parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0),
            WinHttpCloseHandle};
}

bool getText(const std::string& url, std::string& body)
{
    const std::wstring wideUrl = widen(url);
    if (wideUrl.empty())
        return false;
    InternetHandle session{WinHttpOpen(L"ServiceLanguageModelInstaller/1.0",
                                       WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                                       WINHTTP_NO_PROXY_BYPASS, 0),
                           WinHttpCloseHandle};
    if (!session)
        return false;
    InternetHandle request = openRequest(session.get(), wideUrl);
    if (!request || !WinHttpSendRequest(request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                        WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request.get(), nullptr))
        return false;

    DWORD status = 0;
    DWORD statusSize = sizeof(status);
    if (!WinHttpQueryHeaders(request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &statusSize, WINHTTP_NO_HEADER_INDEX) ||
        status != HTTP_STATUS_OK)
        return false;

    std::array<char, 1 << 16> buffer{};
    while (true)
    {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request.get(), &available))
            return false;
        if (available == 0)
            break;
        DWORD received = 0;
        if (!WinHttpReadData(request.get(), buffer.data(),
                             static_cast<DWORD>(std::min<size_t>(buffer.size(), available)), &received) ||
            received == 0)
            return false;
        body.append(buffer.data(), received);
        if (body.size() > 16 * 1024 * 1024)
            return false;
    }
    return true;
}

bool downloadFile(const std::string& url, const fs::path& destination)
{
    const std::wstring wideUrl = widen(url);
    if (wideUrl.empty())
        return false;
    InternetHandle session{WinHttpOpen(L"ServiceLanguageModelInstaller/1.0",
                                       WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                                       WINHTTP_NO_PROXY_BYPASS, 0),
                           WinHttpCloseHandle};
    if (!session)
        return false;
    InternetHandle request = openRequest(session.get(), wideUrl);
    if (!request || !WinHttpSendRequest(request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                        WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request.get(), nullptr))
        return false;

    DWORD status = 0;
    DWORD statusSize = sizeof(status);
    if (!WinHttpQueryHeaders(request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &statusSize, WINHTTP_NO_HEADER_INDEX) ||
        status != HTTP_STATUS_OK)
        return false;

    fs::path partial = destination;
    partial += L".part";
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    if (!output)
        return false;

    std::array<char, 1 << 16> buffer{};
    uint64_t total = 0;
    bool success = true;
    while (true)
    {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request.get(), &available))
        {
            success = false;
            break;
        }
        if (available == 0)
            break;
        DWORD received = 0;
        if (!WinHttpReadData(request.get(), buffer.data(),
                             static_cast<DWORD>(std::min<size_t>(buffer.size(), available)), &received) ||
            received == 0)
        {
            success = false;
            break;
        }
        total += received;
        if (total > MaxAssetBytes)
        {
            success = false;
            break;
        }
        output.write(buffer.data(), received);
        if (!output)
        {
            success = false;
            break;
        }
    }
    output.close();
    success = success && output.good();
    std::error_code ec;
    if (!success)
    {
        fs::remove(partial, ec);
        return false;
    }
    fs::rename(partial, destination, ec);
    return !ec;
}
#else
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
#endif

bool runDesync(const fs::path& executable, const std::vector<std::string>& arguments)
{
#ifdef _WIN32
    auto quote = [](const std::wstring& value) {
        std::wstring result = L"\"";
        size_t slashes = 0;
        for (wchar_t c : value)
        {
            if (c == L'\\')
            {
                ++slashes;
                continue;
            }
            if (c == L'"')
                result.append(slashes * 2 + 1, L'\\');
            else
                result.append(slashes, L'\\');
            result.push_back(c);
            slashes = 0;
        }
        result.append(slashes * 2, L'\\');
        result.push_back(L'"');
        return result;
    };
    std::wstring command = quote(executable.wstring());
    for (const auto& argument : arguments)
        command += L" " + quote(widen(argument));
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                        nullptr, nullptr, &startup, &process))
        return false;
    CloseHandle(process.hThread);
    const DWORD wait = WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exitCode = 1;
    const bool success = wait == WAIT_OBJECT_0 && GetExitCodeProcess(process.hProcess, &exitCode) && exitCode == 0;
    CloseHandle(process.hProcess);
    return success;
#else
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
#endif
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
#ifndef _WIN32
    std::error_code ec;
    fs::permissions(assets / desyncName, fs::perms::owner_exec | fs::perms::group_exec |
                                              fs::perms::others_exec,
                    fs::perm_options::add, ec);
    if (ec)
        return false;
#endif
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
#ifdef _WIN32
    const auto processId = GetCurrentProcessId();
#else
    const auto processId = getpid();
#endif
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
    if (success && !runDesync(desync, {"untar", "--index", "--store", store.string(),
                                       index.string(), target.string()}))
    {
        std::cerr << "Desync could not restore the indexed payload.\n";
        success = false;
    }
    fs::remove_all(store, ec);
#ifndef _WIN32
    // The index is written on Windows, so restore the execute bits of the shipped programs.
    if (success)
        for (const auto& entry : fs::directory_iterator(target / "bin", ec))
            if (entry.is_regular_file())
                fs::permissions(entry.path(), fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                                fs::perm_options::add, ec);
#endif
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
