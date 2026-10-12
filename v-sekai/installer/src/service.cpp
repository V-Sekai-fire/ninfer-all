// SPDX-License-Identifier: MPL-2.0
// ninfer-service.exe: the Windows service host that the MSI installs as the NInfer service.
//
// ninfer-serve.exe is a console program. It does not talk to the Service Control Manager, so this
// host does that. It starts bin\ninfer-serve.exe on models\bonsai2-27b-heretic.ninfer with the
// flags in %ProgramData%\V-Sekai\NInfer\ninfer-serve.args. The child runs in a job object that
// kills it when the host exits, and it writes stdout and stderr to
// %ProgramData%\V-Sekai\NInfer\logs\ninfer-serve.log. The previous log is kept as
// ninfer-serve.log.1. When the child exits with an error, the host starts it again after 10, 30,
// then 60 seconds between attempts, and keeps retrying every 60 seconds. A host without an NVIDIA
// GPU therefore keeps the service running and the log says why ninfer-serve stops. When the child
// exits with 0, the service stops.
// The host restarts the child itself rather than through the SCM failure actions: MSI's
// MsiServiceConfig tables are unreliable for a service installed in the same transaction.
//
// ninfer-serve.args has one option per line, for example "--port 8080". A line holds the flag,
// then optional whitespace and the value. The value is passed as one argument, so a path may
// contain spaces. Blank lines and lines that start with '#' are skipped.
//
//   ninfer-service.exe            run under the Service Control Manager
//   ninfer-service.exe --console  run in the foreground, the same way, until Ctrl+C

#include <windows.h>
#include <shlobj.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace
{
constexpr wchar_t ServiceName[] = L"NInfer";
constexpr wchar_t ModelFile[] = L"bonsai2-27b-heretic.ninfer";

SERVICE_STATUS_HANDLE statusHandle = nullptr;
SERVICE_STATUS status{};
HANDLE stopEvent = nullptr;

void report(DWORD state, DWORD exitCode = NO_ERROR, DWORD waitHint = 0)
{
    status.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    status.dwCurrentState = state;
    status.dwControlsAccepted = state == SERVICE_RUNNING ? SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN : 0;
    // A non-zero child exit is reported as a service-specific error, so the SCM runs the
    // failure actions (restart) the MSI configures.
    status.dwWin32ExitCode = exitCode == NO_ERROR ? NO_ERROR : ERROR_SERVICE_SPECIFIC_ERROR;
    status.dwServiceSpecificExitCode = exitCode;
    status.dwWaitHint = waitHint;
    static DWORD checkPoint = 1;
    status.dwCheckPoint = (state == SERVICE_RUNNING || state == SERVICE_STOPPED) ? 0 : checkPoint++;
    if (statusHandle)
        SetServiceStatus(statusHandle, &status);
}

fs::path moduleDir()
{
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    buffer.resize(length);
    return fs::path(buffer).parent_path();
}

fs::path dataDir()
{
    PWSTR programData = nullptr;
    fs::path result = L"C:\\ProgramData";
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_ProgramData, 0, nullptr, &programData)))
        result = programData;
    CoTaskMemFree(programData);
    return result / L"V-Sekai" / L"NInfer";
}

std::wstring widen(const std::string& text)
{
    if (text.empty())
        return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring result(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), size);
    return result;
}

std::string narrow(const std::wstring& text)
{
    if (text.empty())
        return {};
    const int size =
        WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
    std::string result(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), size, nullptr, nullptr);
    return result;
}

std::wstring trim(const std::wstring& text)
{
    const auto first = text.find_first_not_of(L" \t\r\n");
    if (first == std::wstring::npos)
        return {};
    const auto last = text.find_last_not_of(L" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::vector<std::wstring> readArgs(const fs::path& file)
{
    std::vector<std::wstring> args;
    std::ifstream input(file, std::ios::binary);
    std::string line;
    bool first = true;
    while (std::getline(input, line))
    {
        if (first && line.rfind("\xEF\xBB\xBF", 0) == 0)
            line.erase(0, 3);
        first = false;
        const std::wstring text = trim(widen(line));
        if (text.empty() || text[0] == L'#')
            continue;
        const auto gap = text.find_first_of(L" \t");
        if (gap == std::wstring::npos)
        {
            args.push_back(text);
            continue;
        }
        args.push_back(text.substr(0, gap));
        args.push_back(trim(text.substr(gap)));
    }
    return args;
}

// CommandLineToArgvW quoting, so every argument reaches the child unchanged.
std::wstring quote(const std::wstring& arg)
{
    if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring::npos)
        return arg;
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (wchar_t c : arg)
    {
        if (c == L'\\')
        {
            ++slashes;
            continue;
        }
        result.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        slashes = 0;
        result.push_back(c);
    }
    result.append(slashes * 2, L'\\');
    result.push_back(L'"');
    return result;
}

void rotateLog(const fs::path& log)
{
    std::error_code ignored;
    if (fs::exists(log, ignored))
        fs::rename(log, fs::path(log).concat(L".1"), ignored);
}

// Start the child once and wait for it or for a stop request. Returns the child's exit code, or
// 0 after a requested stop.
DWORD runOnce()
{
    const fs::path root = moduleDir().parent_path();
    const fs::path exe = root / L"bin" / L"ninfer-serve.exe";
    const fs::path model = root / L"models" / ModelFile;
    const fs::path data = dataDir();
    const fs::path logs = data / L"logs";
    std::error_code ignored;
    fs::create_directories(logs, ignored);
    const fs::path log = logs / L"ninfer-serve.log";
    rotateLog(log);

    std::wstring commandLine = quote(exe.wstring()) + L" " + quote(model.wstring());
    for (const auto& arg : readArgs(data / L"ninfer-serve.args"))
        commandLine += L" " + quote(arg);

    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE output = CreateFileW(log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (output == INVALID_HANDLE_VALUE)
        return GetLastError();
    const std::string header = "ninfer-service: " + narrow(commandLine) + "\r\n";
    DWORD written = 0;
    WriteFile(output, header.data(), static_cast<DWORD>(header.size()), &written, nullptr);

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = output;
    startup.hStdError = output;
    PROCESS_INFORMATION process{};
    const BOOL started = CreateProcessW(exe.c_str(), commandLine.data(), nullptr, nullptr, TRUE,
                                        CREATE_SUSPENDED | CREATE_NO_WINDOW, nullptr, root.c_str(), &startup,
                                        &process);
    CloseHandle(output);
    if (!started)
    {
        const DWORD error = GetLastError();
        CloseHandle(job);
        return error;
    }
    AssignProcessToJobObject(job, process.hProcess);
    ResumeThread(process.hThread);
    CloseHandle(process.hThread);

    HANDLE waits[] = {process.hProcess, stopEvent};
    const DWORD which = WaitForMultipleObjects(2, waits, FALSE, INFINITE);
    DWORD exitCode = 0;
    if (which == WAIT_OBJECT_0)
        GetExitCodeProcess(process.hProcess, &exitCode);
    else
        TerminateJobObject(job, 0);
    CloseHandle(process.hProcess);
    CloseHandle(job);
    return exitCode;
}

// Run the child until it exits with 0 or a stop is requested, restarting it after an error.
DWORD run(bool console)
{
    constexpr DWORD backoffMs[] = {10000, 30000, 60000};
    if (!console)
        report(SERVICE_RUNNING);
    for (size_t attempt = 0;; ++attempt)
    {
        const DWORD exitCode = runOnce();
        if (exitCode == 0 || WaitForSingleObject(stopEvent, 0) == WAIT_OBJECT_0)
            return exitCode;
        const DWORD delay = backoffMs[attempt < 2 ? attempt : 2];
        std::wcerr << L"ninfer-serve exited with " << exitCode << L"; restarting in " << delay / 1000 << L" s\n";
        if (WaitForSingleObject(stopEvent, delay) == WAIT_OBJECT_0)
            return 0;
    }
}

DWORD WINAPI control(DWORD code, DWORD, void*, void*)
{
    if (code == SERVICE_CONTROL_STOP || code == SERVICE_CONTROL_SHUTDOWN)
    {
        report(SERVICE_STOP_PENDING, NO_ERROR, 10000);
        SetEvent(stopEvent);
        return NO_ERROR;
    }
    return code == SERVICE_CONTROL_INTERROGATE ? NO_ERROR : ERROR_CALL_NOT_IMPLEMENTED;
}

void WINAPI serviceMain(DWORD, LPWSTR*)
{
    statusHandle = RegisterServiceCtrlHandlerExW(ServiceName, control, nullptr);
    if (!statusHandle)
        return;
    report(SERVICE_START_PENDING, NO_ERROR, 10000);
    const DWORD exitCode = run(false);
    report(SERVICE_STOPPED, exitCode);
}

BOOL WINAPI consoleControl(DWORD)
{
    SetEvent(stopEvent);
    return TRUE;
}
} // namespace

int wmain(int argc, wchar_t** argv)
{
    stopEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (argc > 1 && std::wstring(argv[1]) == L"--console")
    {
        SetConsoleCtrlHandler(consoleControl, TRUE);
        const DWORD exitCode = run(true);
        std::wcerr << L"ninfer-serve exited with " << exitCode << L"\n";
        return static_cast<int>(exitCode);
    }
    SERVICE_TABLE_ENTRYW table[] = {{const_cast<LPWSTR>(ServiceName), serviceMain}, {nullptr, nullptr}};
    if (!StartServiceCtrlDispatcherW(table))
    {
        std::wcerr << L"Start this program as the NInfer service, or pass --console.\n";
        return static_cast<int>(GetLastError());
    }
    return 0;
}
