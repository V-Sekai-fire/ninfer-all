# SPDX-License-Identifier: MPL-2.0
# The desync release installers and volume packer, ported from
# V-Sekai-fire/service-language-model tray/CMakeLists.txt.
# Windows: ninfer-setup (shipped as setup.exe), payload-pack, and ninfer-service (the MSI's
# Windows service host).
# Linux: ninfer-install, which restores payload-linux.caidx from the same volumes, and
# payload-pack, so a Linux host can make a Linux-only release.
cmake_minimum_required(VERSION 3.24)
project(ninfer_installer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NI_RELEASE_TAG "" CACHE STRING "GitHub release tag embedded in setup.exe")
if(NOT NI_RELEASE_TAG STREQUAL "" AND NOT NI_RELEASE_TAG MATCHES "^(v[0-9]+\\.[0-9]+\\.[0-9]+-)?dev\\.[A-Za-z0-9.-]+$")
    message(FATAL_ERROR "NI_RELEASE_TAG must be empty or a dev.* or vX.Y.Z-dev.* tag")
endif()
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_executable(payload-pack src/pack.cpp)

if(WIN32)
    add_executable(ninfer-setup WIN32 src/install.cpp)
    target_compile_definitions(ninfer-setup PRIVATE "NI_RELEASE_TAG=L\"${NI_RELEASE_TAG}\"")
    target_link_libraries(ninfer-setup PRIVATE shell32 winhttp)
    target_link_options(ninfer-setup PRIVATE "/MANIFESTUAC:level='requireAdministrator'")
    # The service host the MSI installs as the NInfer Windows service.
    add_executable(ninfer-service src/service.cpp)
    target_link_libraries(ninfer-service PRIVATE shell32 ole32)
    if(MSVC)
        target_compile_options(ninfer-setup PRIVATE /W4 /permissive-)
        target_compile_options(payload-pack PRIVATE /W4 /permissive-)
        target_compile_options(ninfer-service PRIVATE /W4 /permissive-)
    endif()
else()
    add_executable(ninfer-install src/install-portable.cpp)
    target_link_options(ninfer-install PRIVATE -static-libstdc++ -static-libgcc)
endif()
