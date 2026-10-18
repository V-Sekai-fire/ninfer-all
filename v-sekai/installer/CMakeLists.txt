# SPDX-License-Identifier: MPL-2.0
# The desync release installer and volume packer, ported from
# V-Sekai-fire/service-language-model tray/CMakeLists.txt. Linux only:
# ninfer-install restores payload-linux.caidx from the payload-data volumes,
# and payload-pack writes those volumes from a desync chunk store.
cmake_minimum_required(VERSION 3.24)
project(ninfer_installer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(payload-pack src/pack.cpp)
add_executable(ninfer-install src/install-portable.cpp)
target_link_options(ninfer-install PRIVATE -static-libstdc++ -static-libgcc)
