// SPDX-License-Identifier: MPL-2.0
// Ported from V-Sekai-fire/service-language-model tray/src/install.cpp at 2365a9d0.

#include <windows.h>
#include <shellapi.h>
#include <winhttp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#ifndef NI_RELEASE_TAG
#define NI_RELEASE_TAG L""
#endif

namespace fs = std::filesystem;
namespace
{
constexpr char Magic[8] = {'D', 'S', 'B', 'L', 'O', 'B', '1', '\n'};
constexpr wchar_t ReleaseTag[] = NI_RELEASE_TAG;
constexpr uint64_t MaxAssetBytes = 1900000000;
using InternetHandle = std::unique_ptr<void, decltype(&WinHttpCloseHandle)>;

void notify(const std::wstring& message, const wchar_t* title, UINT flags, bool quiet)
{
    if (!quiet)
        MessageBoxW(nullptr, message.c_str(), title, MB_OK | flags);
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

std::wstring quote(const std::wstring& value)
{
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
}

bool runDesync(const fs::path& executable, const std::vector<std::wstring>& args, DWORD& exitCode)
{
    std::wstring command = quote(executable.wstring());
    for (const auto& arg : args)
        command += L" " + quote(arg);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr,
                        &startup, &process))
        return false;
    CloseHandle(process.hThread);
    const DWORD wait = WaitForSingleObject(process.hProcess, INFINITE);
    const bool gotCode = wait == WAIT_OBJECT_0 && GetExitCodeProcess(process.hProcess, &exitCode);
    CloseHandle(process.hProcess);
    return gotCode && exitCode == 0;
}

enum class DownloadResult
{
    Downloaded,
    NotFound,
    Failed
};

DownloadResult downloadReleaseAsset(HINTERNET session, const std::wstring& name, const fs::path& destination,
                                    std::wstring& error)
{
    InternetHandle connection{WinHttpConnect(session, L"github.com", INTERNET_DEFAULT_HTTPS_PORT, 0),
                              WinHttpCloseHandle};
    if (!connection)
    {
        error = L"Cannot connect to GitHub (WinHTTP error " + std::to_wstring(GetLastError()) + L").";
        return DownloadResult::Failed;
    }

    std::wstring path = L"/V-Sekai-fire/ninfer-all/releases/download/";
    path += ReleaseTag;
    path += L"/";
    path += name;
    InternetHandle request{WinHttpOpenRequest(connection.get(), L"GET", path.c_str(), nullptr,
                                              WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                              WINHTTP_FLAG_SECURE),
                           WinHttpCloseHandle};
    if (!request || !WinHttpSendRequest(request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                        WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request.get(), nullptr))
    {
        error = L"Cannot request " + name + L" from GitHub (WinHTTP error " + std::to_wstring(GetLastError()) + L").";
        return DownloadResult::Failed;
    }

    DWORD status = 0;
    DWORD statusSize = sizeof(status);
    if (!WinHttpQueryHeaders(request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &statusSize, WINHTTP_NO_HEADER_INDEX))
    {
        error = L"Cannot read the GitHub response for " + name + L" (WinHTTP error " +
                std::to_wstring(GetLastError()) + L").";
        return DownloadResult::Failed;
    }
    if (status == HTTP_STATUS_NOT_FOUND)
        return DownloadResult::NotFound;
    if (status != HTTP_STATUS_OK)
    {
        error = L"GitHub returned HTTP " + std::to_wstring(status) + L" while downloading " + name + L".";
        return DownloadResult::Failed;
    }

    DWORD contentLength = 0;
    DWORD contentLengthSize = sizeof(contentLength);
    const bool hasContentLength =
        WinHttpQueryHeaders(request.get(), WINHTTP_QUERY_CONTENT_LENGTH | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &contentLength, &contentLengthSize,
                            WINHTTP_NO_HEADER_INDEX) != FALSE;
    if (hasContentLength && contentLength > MaxAssetBytes)
    {
        error = L"GitHub asset exceeds the supported size limit: " + name;
        return DownloadResult::Failed;
    }

    fs::path partial = destination;
    partial += L".part";
    std::error_code ec;
    fs::remove(partial, ec);
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        error = L"Cannot create downloaded asset: " + partial.wstring();
        return DownloadResult::Failed;
    }

    std::array<char, 1 << 16> buffer{};
    uint64_t received = 0;
    while (true)
    {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request.get(), &available))
        {
            error = L"Download interrupted for " + name + L" (WinHTTP error " +
                    std::to_wstring(GetLastError()) + L").";
            break;
        }
        if (available == 0)
            break;

        while (available > 0)
        {
            DWORD read = 0;
            const DWORD requested = static_cast<DWORD>(std::min<size_t>(available, buffer.size()));
            if (!WinHttpReadData(request.get(), buffer.data(), requested, &read) || read == 0)
            {
                error = L"Download interrupted for " + name + L" (WinHTTP error " +
                        std::to_wstring(GetLastError()) + L").";
                break;
            }
            if (received + read > MaxAssetBytes)
            {
                error = L"GitHub asset exceeds the supported size limit: " + name;
                break;
            }
            output.write(buffer.data(), read);
            if (!output)
            {
                error = L"Disk write failed while downloading " + name + L".";
                break;
            }
            received += read;
            available -= read;
        }
        if (!error.empty())
            break;
    }

    output.close();
    if (error.empty() && !output)
        error = L"Disk write failed while finalizing " + name + L".";
    if (error.empty() && hasContentLength && received != contentLength)
        error = L"GitHub sent an incomplete asset: " + name;
    if (!error.empty())
    {
        fs::remove(partial, ec);
        return DownloadResult::Failed;
    }

    fs::rename(partial, destination, ec);
    if (ec)
    {
        fs::remove(partial, ec);
        error = L"Cannot finalize downloaded asset: " + destination.wstring();
        return DownloadResult::Failed;
    }
    return DownloadResult::Downloaded;
}

bool downloadRelease(const fs::path& destination, std::wstring& error, bool quiet)
{
    if (ReleaseTag[0] == L'\0')
    {
        error = L"This setup executable is not pinned to a release tag. Use the complete offline bundle.";
        return false;
    }

    notify(L"Downloading the Windows payload from GitHub. This release contains several gigabytes of data.",
           L"Online installer", MB_ICONINFORMATION, quiet);

    InternetHandle session{WinHttpOpen(L"NInferSetup/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                       WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0),
                           WinHttpCloseHandle};
    if (!session)
    {
        error = L"Cannot initialize HTTPS downloads (WinHTTP error " + std::to_wstring(GetLastError()) + L").";
        return false;
    }
    if (!WinHttpSetTimeouts(session.get(), 10000, 30000, 30000, 30000))
    {
        error = L"Cannot configure HTTPS download timeouts (WinHTTP error " +
                std::to_wstring(GetLastError()) + L").";
        return false;
    }

    for (const std::wstring& name : {L"desync.exe", L"payload.caidx"})
    {
        if (downloadReleaseAsset(session.get(), name, destination / name, error) != DownloadResult::Downloaded)
        {
            if (error.empty())
                error = L"The GitHub release is missing required asset " + name + L".";
            return false;
        }
    }

    for (unsigned int i = 0; i < 1000; ++i)
    {
        wchar_t name[32]{};
        swprintf_s(name, L"payload-data-%03u.bin", i);
        const DownloadResult result = downloadReleaseAsset(session.get(), name, destination / name, error);
        if (result == DownloadResult::NotFound)
        {
            if (i == 0)
            {
                error = L"The GitHub release does not contain any payload data volumes.";
                return false;
            }
            return true;
        }
        if (result == DownloadResult::Failed)
            return false;
    }

    error = L"The GitHub release has more data volumes than this installer supports.";
    return false;
}

bool validChunkPath(const std::string& name)
{
    if (name.empty() || name.front() == '/' || name.find('\\') != std::string::npos ||
        name.find(':') != std::string::npos || name.find('\0') != std::string::npos)
        return false;
    std::filesystem::path path(name);
    if (path.is_absolute())
        return false;
    for (const auto& part : path)
        if (part == "." || part == "..")
            return false;
    return true;
}

bool unpackVolume(const fs::path& volume, const fs::path& store, std::wstring& error)
{
    std::ifstream input(volume, std::ios::binary);
    std::array<char, sizeof(Magic)> magic{};
    uint64_t entries = 0;
    if (!input.read(magic.data(), magic.size()) ||     magic != std::array<char, sizeof(Magic)>{'D', 'S', 'B', 'L', 'O', 'B', '1', '\n'} ||
        !readU64(input, entries))
    {
        error = L"Invalid or truncated data volume: " + volume.filename().wstring();
        return false;
    }

    for (uint64_t i = 0; i < entries; ++i)
    {
        uint32_t nameSize = 0;
        uint64_t fileSize = 0;
        if (!readU32(input, nameSize) || !readU64(input, fileSize) || nameSize == 0 || nameSize > 4096)
        {
            error = L"Invalid entry in data volume: " + volume.filename().wstring();
            return false;
        }
        std::string name(nameSize, '\0');
        if (!input.read(name.data(), nameSize) || !validChunkPath(name))
        {
            error = L"Invalid chunk path in data volume: " + volume.filename().wstring();
            return false;
        }

        const fs::path destination = store / fs::path(name);
        std::error_code ec;
        fs::create_directories(destination.parent_path(), ec);
        if (ec || fs::exists(destination))
        {
            error = L"Cannot create unique chunk file from " + volume.filename().wstring();
            return false;
        }
        std::ofstream output(destination, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            error = L"Cannot write desync chunk: " + destination.wstring();
            return false;
        }
        std::array<char, 1 << 16> buffer{};
        uint64_t remaining = fileSize;
        while (remaining > 0)
        {
            const auto count = static_cast<std::streamsize>(std::min<uint64_t>(remaining, buffer.size()));
            if (!input.read(buffer.data(), count))
            {
                error = L"Truncated chunk in data volume: " + volume.filename().wstring();
                return false;
            }
            output.write(buffer.data(), count);
            if (!output)
            {
                error = L"Disk write failed while extracting: " + volume.filename().wstring();
                return false;
            }
            remaining -= static_cast<uint64_t>(count);
        }
    }

    if (input.peek() != std::char_traits<char>::eof())
    {
        error = L"Unexpected trailing bytes in data volume: " + volume.filename().wstring();
        return false;
    }
    return true;
}

bool hasCompleteAssetBundle(const fs::path& assets)
{
    std::error_code ec;
    if (!fs::is_regular_file(assets / L"desync.exe", ec) || ec ||
        !fs::is_regular_file(assets / L"payload.caidx", ec) || ec)
        return false;

    std::vector<fs::path> volumes;
    const std::wregex pattern(LR"(payload-data-([0-9]{3})\.bin)");
    fs::directory_iterator entry(assets, ec);
    const fs::directory_iterator end;
    for (; !ec && entry != end; entry.increment(ec))
    {
        if (entry->is_regular_file(ec) && !ec &&
            std::regex_match(entry->path().filename().wstring(), pattern))
            volumes.push_back(entry->path());
    }
    if (ec || volumes.empty())
        return false;

    std::sort(volumes.begin(), volumes.end());
    for (size_t i = 0; i < volumes.size(); ++i)
    {
        wchar_t expected[32]{};
        swprintf_s(expected, L"payload-data-%03zu.bin", i);
        if (volumes[i].filename().wstring() != expected)
            return false;
    }
    return true;
}

// The server binary marks a real install directory, so uninstall never deletes an unrelated folder.
constexpr wchar_t ServerMarker[] = L"bin\\ninfer-serve.exe";

int uninstall(const fs::path& target, bool quiet)
{
    if (!fs::exists(target))
    {
        notify(L"NInfer is already absent.", L"Uninstaller", MB_ICONINFORMATION, quiet);
        return 0;
    }
    if (!fs::is_regular_file(target / ServerMarker))
    {
        notify(L"The install directory does not contain bin\\ninfer-serve.exe. Files were left untouched:\n" +
                   target.wstring(),
               L"Uninstaller", MB_ICONERROR, quiet);
        return 1;
    }

    std::error_code ec;
    fs::remove_all(target, ec);
    if (ec)
    {
        notify(L"The NInfer files could not be deleted. Stop ninfer-serve.exe and retry:\n" + target.wstring(),
               L"Uninstaller", MB_ICONERROR, quiet);
        return 1;
    }
    notify(L"NInfer uninstalled.", L"Uninstaller", MB_ICONINFORMATION, quiet);
    return 0;
}

int install(const fs::path& assets, const fs::path& target, bool quiet)
{
    const fs::path desync = assets / L"desync.exe";
    const fs::path index = assets / L"payload.caidx";
    if (!fs::is_regular_file(desync) || !fs::is_regular_file(index))
    {
        notify(L"Required setup assets are missing. Keep setup.exe, desync.exe, payload.caidx, and every "
               L"payload-data volume together for offline installation, or run setup.exe --online.",
               L"Installer: missing files", MB_ICONERROR, quiet);
        return 1;
    }

    std::vector<fs::path> volumes;
    const std::wregex pattern(LR"(payload-data-([0-9]{3})\.bin)");
    for (const auto& entry : fs::directory_iterator(assets))
    {
        std::wsmatch match;
        const std::wstring name = entry.path().filename().wstring();
        if (entry.is_regular_file() && std::regex_match(name, match, pattern))
            volumes.push_back(entry.path());
    }
    std::sort(volumes.begin(), volumes.end());
    if (volumes.empty())
    {
        notify(L"No payload-data-*.bin volumes were found beside setup.exe.",
               L"Installer: missing data", MB_ICONERROR, quiet);
        return 1;
    }
    for (size_t i = 0; i < volumes.size(); ++i)
    {
        wchar_t expected[32]{};
        swprintf_s(expected, L"payload-data-%03zu.bin", i);
        if (volumes[i].filename().wstring() != expected)
        {
            notify(L"The data-volume set is incomplete or out of sequence.",
                   L"Installer: incomplete data", MB_ICONERROR, quiet);
            return 1;
        }
    }

    wchar_t temp[MAX_PATH + 1]{};
    const DWORD tempLength = GetTempPathW(MAX_PATH, temp);
    if (tempLength == 0 || tempLength > MAX_PATH)
    {
        notify(L"Cannot find the temporary directory.", L"Installer", MB_ICONERROR, quiet);
        return 1;
    }
    const fs::path store = fs::path(temp) /
                           (L"ninfer-all-" + std::to_wstring(GetCurrentProcessId()));
    std::error_code ec;
    if (!fs::create_directory(store, ec) || ec)
    {
        notify(L"Cannot create a temporary directory for the payload.", L"Installer",
               MB_ICONERROR, quiet);
        return 1;
    }

    int result = 1;
    do
    {
        std::wstring error;
        for (const auto& volume : volumes)
        {
            if (!unpackVolume(volume, store, error))
                break;
        }
        if (!error.empty())
        {
            notify(error, L"Installer: invalid data", MB_ICONERROR, quiet);
            break;
        }

        std::error_code createError;
        fs::create_directories(target, createError);
        if (createError)
        {
            notify(L"Cannot create install directory:\n" + target.wstring(), L"Installer",
                   MB_ICONERROR, quiet);
            break;
        }

        DWORD exitCode = 1;
        if (!runDesync(desync, {L"verify", L"--store", store.wstring()}, exitCode))
        {
            const std::wstring message = L"Desync rejected the reconstructed chunk store. Exit code: " +
                                         std::to_wstring(exitCode);
            notify(message, L"Installer: data verification failed", MB_ICONERROR, quiet);
            break;
        }
        if (!runDesync(desync, {L"untar", L"--index", L"--store", store.wstring(), index.wstring(),
                                target.wstring()},
                       exitCode))
        {
            const std::wstring message = L"Desync could not restore the offline payload. Exit code: " +
                                         std::to_wstring(exitCode);
            notify(message, L"Installer: restore failed", MB_ICONERROR, quiet);
            break;
        }
        if (!fs::is_regular_file(target / ServerMarker))
        {
            notify(L"The restored payload has no bin\\ninfer-serve.exe.", L"Installer: restore failed",
                   MB_ICONERROR, quiet);
            break;
        }
        notify(L"Installation complete. Start the server with serve.nu in:\n" + target.wstring(),
               L"Installer", MB_ICONINFORMATION, quiet);
        result = 0;
    } while (false);

    fs::remove_all(store, ec);
    return result;
}
} // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
{
    std::vector<wchar_t> module(32768);
    const DWORD length = GetModuleFileNameW(nullptr, module.data(), static_cast<DWORD>(module.size()));
    if (length == 0 || length == module.size())
    {
        MessageBoxW(nullptr, L"Cannot determine setup.exe location.", L"Installer", MB_OK | MB_ICONERROR);
        return 1;
    }
    fs::path assets = fs::path(std::wstring(module.data(), length)).parent_path();
    wchar_t* programData = nullptr;
    size_t programDataLength = 0;
    if (_wdupenv_s(&programData, &programDataLength, L"ProgramData") != 0 || programData == nullptr)
    {
        MessageBoxW(nullptr, L"Cannot determine the shared ProgramData folder.", L"Installer",
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    fs::path target = fs::path(programData) / L"V-Sekai" / L"NInfer";
    free(programData);

    bool quiet = false;
    int argumentCount = 0;
    wchar_t** arguments = CommandLineToArgvW(GetCommandLineW(), &argumentCount);
    if (arguments == nullptr)
    {
        MessageBoxW(nullptr, L"Cannot read setup arguments.", L"Installer", MB_OK | MB_ICONERROR);
        return 1;
    }
    bool online = false;
    bool offline = false;
    bool removeInstall = false;
    bool fromSpecified = false;
    bool targetSpecified = false;
    for (int i = 1; i < argumentCount; ++i)
    {
        const std::wstring option = arguments[i];
        if (option == L"--quiet")
            quiet = true;
        else if (option == L"--online")
            online = true;
        else if (option == L"--offline")
            offline = true;
        else if (option == L"--uninstall")
            removeInstall = true;
        else if ((option == L"--from" || option == L"--target") && i + 1 < argumentCount)
        {
            const fs::path value = arguments[++i];
            (option == L"--from" ? assets : target) = value;
            if (option == L"--from")
                fromSpecified = true;
            else
                targetSpecified = true;
        }
        else
        {
            LocalFree(arguments);
            MessageBoxW(nullptr, L"Usage: setup.exe [--online | --offline] [--from <asset-folder>] "
                                 L"[--target <folder>] [--uninstall] [--quiet]",
                        L"Installer", MB_OK | MB_ICONERROR);
            return 2;
        }
    }
    LocalFree(arguments);

    if ((online && offline) || (online && fromSpecified) ||
        (removeInstall && (online || offline || fromSpecified || targetSpecified)))
    {
        notify(L"Choose one install mode, or run --uninstall by itself.", L"Installer", MB_ICONERROR, quiet);
        return 2;
    }

    if (removeInstall)
        return uninstall(fs::absolute(target), quiet);

    fs::path downloadedAssets;
    if (online || (!offline && !fromSpecified && !hasCompleteAssetBundle(assets)))
    {
        if (!online)
        {
            if (quiet)
            {
                notify(L"Offline assets are incomplete. Use setup.exe --online to download them.",
                       L"Installer: missing files", MB_ICONERROR, quiet);
                return 1;
            }
            const int choice = MessageBoxW(
                nullptr, L"The complete offline payload is not beside setup.exe.\n\n"
                         L"Download the payload from GitHub now? This release contains several gigabytes of data.",
                L"Online installer", MB_YESNO | MB_ICONQUESTION);
            if (choice != IDYES)
                return 0;
        }

        wchar_t temp[MAX_PATH + 1]{};
        const DWORD tempLength = GetTempPathW(MAX_PATH, temp);
        if (tempLength == 0 || tempLength > MAX_PATH)
        {
            notify(L"Cannot find the temporary directory.", L"Installer", MB_ICONERROR, quiet);
            return 1;
        }
        downloadedAssets = fs::path(temp) /
                           (L"ninfer-all-download-" + std::to_wstring(GetCurrentProcessId()));
        std::error_code ec;
        if (!fs::create_directory(downloadedAssets, ec) || ec)
        {
            notify(L"Cannot create a temporary download directory.", L"Installer", MB_ICONERROR, quiet);
            return 1;
        }

        std::wstring error;
        if (!downloadRelease(downloadedAssets, error, quiet))
        {
            notify(error, L"Installer: download failed", MB_ICONERROR, quiet);
            fs::remove_all(downloadedAssets, ec);
            return 1;
        }
        assets = downloadedAssets;
    }

    const int result = install(fs::absolute(assets), fs::absolute(target), quiet);
    if (!downloadedAssets.empty())
    {
        std::error_code ec;
        fs::remove_all(downloadedAssets, ec);
    }
    return result;
}
