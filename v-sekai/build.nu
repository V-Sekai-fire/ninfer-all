# Build ninfer-serve and ninfer-calibrate for one RTX 4090 (sm_89), on Windows or Linux,
# against the pixi toolchain, and stage them with their library closure in dist/<os>/.
# Run from the fork root: pixi run --locked -e build nu v-sekai/build.nu
#
# Windows: MSVC comes from Visual Studio 2022. FindFFMPEG.cmake reads a vcpkg-shaped tree,
# so a junction maps .deps/installed/x64-windows onto the pixi environment's Library directory.
# Linux: GCC 14 and a glibc 2.28 sysroot come from pixi. pkg-config finds FFmpeg and libcurl.
# Each staged binary loads its libraries from ../lib through an $ORIGIN rpath.

use msvc.nu load-msvc

const programs = [ninfer-serve ninfer-calibrate]

# Reproducibility: the pixi lockfile pins CMake, Ninja, CUDA, GCC, the sysroot and every library.
# Visual Studio is outside pixi, so the MSVC toolset is pinned here and checked.
const msvc_toolset = "14.44.35207"

# Append to a flags variable that a conda activation script may already have set.
def --env add-flags [name: string, flags: string] {
    let current = ($env | get -o $name | default "")
    load-env {($name): ($"($current) ($flags)" | str trim)}
}

def check [what: string] {
    if $env.LAST_EXIT_CODE != 0 { error make {msg: $"($what) failed with exit code ($env.LAST_EXIT_CODE)"} }
}

def --env configure-windows [root: path, prefix: path] {
    let library = ($prefix | path join Library)
    let link = ($root | path join .deps installed x64-windows)
    mkdir ($link | path dirname)
    if not ($link | path exists) {
        ^cmd /c mklink /J ($link | str replace -a '/' '\') ($library | str replace -a '/' '\')
    }
    load-msvc
    if $env.VCToolsVersion? != $msvc_toolset {
        error make {msg: $"MSVC toolset ($env.VCToolsVersion?) is not the pinned ($msvc_toolset)"}
    }
    # /Brepro drops timestamps from objects and PE headers; /d1trimfile strips the checkout path.
    let trim = $"/d1trimfile:($root | str replace -a '/' '\\')\\"
    add-flags CFLAGS $"/Brepro ($trim)"
    add-flags CXXFLAGS $"/Brepro ($trim)"
    add-flags CUDAFLAGS $"-Xcompiler=/Brepro -Xcompiler=($trim)"
    add-flags LDFLAGS "/Brepro"
    $env.VCPKG_ROOT = ($root | path join .deps)
    $env.VCPKG_TARGET_TRIPLET = "x64-windows"
    $env.CUDA_PATH = $library
    $env.CUDACXX = ($library | path join bin nvcc.exe)
    [$"-DCMAKE_PREFIX_PATH=($link)" $"-DCUDAToolkit_ROOT=($library)"]
}

def --env configure-linux [root: path, prefix: path] {
    # WSL appends the Windows PATH as /mnt/<drive> entries. CMake's package search would walk
    # them over the 9p bridge, which takes many minutes per lookup, so drop them.
    $env.PATH = ($env.PATH | where {|d| not ($d | str starts-with /mnt/) })
    $env.CUDACXX = ($prefix | path join bin nvcc)
    # Map the checkout path out of debug info, __FILE__ and assert strings.
    let map = $"-ffile-prefix-map=($root)=."
    add-flags CFLAGS $map
    add-flags CXXFLAGS $map
    add-flags CUDAFLAGS $"-Xcompiler=($map)"
    $env.PKG_CONFIG_PATH = ($prefix | path join lib pkgconfig)
    [$"-DCUDAToolkit_ROOT=($prefix)" $"-DCMAKE_CUDA_HOST_COMPILER=($env.CXX)" $"-DCMAKE_PREFIX_PATH=($prefix)"]
}

# Windows: copy every DLL the programs import, transitively, from the pixi Library\bin.
def stage-windows [bin: path, prefix: path] {
    let library_bin = ($prefix | path join Library bin)
    mut queue = (ls ($bin | path join "*.exe" | into glob) | get name)
    mut seen = []
    while ($queue | is-not-empty) {
        let file = ($queue | first)
        $queue = ($queue | skip 1)
        let deps = (^dumpbin /nologo /dependents $file | lines | str trim | where {|l| $l =~ '(?i)\.dll$' })
        for dll in $deps {
            let key = ($dll | str lowercase)
            if $key in $seen { continue }
            $seen = ($seen | append $key)
            let source = ($library_bin | path join $dll)
            if ($source | path exists) {
                cp $source $bin
                $queue = ($queue | append ($bin | path join $dll))
            }
        }
    }
}

# Linux: copy every shared library that resolves into the pixi prefix, then point rpaths at it.
# The driver (libcuda) and glibc come from the host.
def stage-linux [dist: path, prefix: path] {
    let bin = ($dist | path join bin)
    let lib = ($dist | path join lib)
    mkdir $lib
    let exes = (ls ($bin | path join "*" | into glob) | get name)
    let libs = ($exes | each {|exe|
        ^ldd $exe | lines | parse -r '=> (?<path>/\S+)' | get path
    } | flatten | uniq | where {|p| $p | str starts-with $prefix })
    for l in $libs { ^cp -L $l $lib }
    for exe in $exes {
        ^patchelf --set-rpath '$ORIGIN/../lib' $exe
        ^strip --strip-unneeded $exe
    }
    for so in (ls ($lib | path join "*" | into glob) | get name) {
        ^patchelf --set-rpath '$ORIGIN' $so
    }
    # Prove the closure: every library must resolve, and none from the build prefix.
    for exe in $exes {
        let out = (^ldd $exe | lines)
        let missing = ($out | where {|l| ($l =~ 'not found') and ($l !~ 'libcuda\.so') })
        let leaked = ($out | where {|l| $l =~ $prefix })
        if ($missing | is-not-empty) or ($leaked | is-not-empty) {
            error make {msg: $"Library closure of ($exe) is incomplete: ($missing ++ $leaked)"}
        }
    }
}

# Record what produced the binaries, and hash every staged file. Two builds of one commit
# must produce identical SHA256SUMS files.
def write-buildinfo [root: path, dist: path, os: string, arch: string] {
    let nvcc = (^$env.CUDACXX --version | lines | last 1 | str trim)
    let cxx = if $os == windows { $"MSVC ($env.VCToolsVersion)" } else { ^$env.CXX --version | lines | first }
    {
        commit: (^git rev-parse HEAD | str trim)
        source_date_epoch: $env.SOURCE_DATE_EPOCH
        pixi_lock_sha256: (open --raw ($root | path join pixi.lock) | hash sha256)
        cuda_arch: $arch
        nvcc: $nvcc
        cxx: $cxx
        cmake: (^cmake --version | lines | first)
    } | to json | save -f ($dist | path join BUILDINFO.json)
    let sums = (glob ($dist | path join "**" "*") --no-dir
        | where {|f| ($f | path basename) not-in [SHA256SUMS BUILDINFO.json] }
        | sort
        | each {|f| $"(open --raw $f | hash sha256)  ($f | path relative-to $dist | str replace -a '\\' '/')" })
    $sums | str join "\n" | save -f ($dist | path join SHA256SUMS)
    print $"SHA256SUMS: (open --raw ($dist | path join SHA256SUMS) | hash sha256)"
}

def main [--arch: string = "89", --build-dir: string = "build"] {
    let root = ($env.FILE_PWD | path dirname)
    cd $root
    let os = $nu.os-info.name
    let prefix = $env.CONDA_PREFIX
    let build_dir = ($root | path join $"($build_dir)-($os)")

    # Build only committed sources with the committed lockfile.
    let dirty = (^git status --porcelain --untracked-files=no | str trim)
    if ($dirty | is-not-empty) { error make {msg: $"Commit or revert these changes first:\n($dirty)"} }
    $env.SOURCE_DATE_EPOCH = (^git log -1 --format=%ct | str trim)
    let extra = if $os == windows { configure-windows $root $prefix } else { configure-linux $root $prefix }

    # --fresh rebuilds the cache, so changed flags in the environment always take effect.
    (^cmake --fresh -S . -B $build_dir -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        $"-DCMAKE_CUDA_ARCHITECTURES=($arch)"
        -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
        ...$extra)
    check "cmake configure"
    ^cmake --build $build_dir --target ...$programs
    check "cmake build"

    let dist = ($root | path join dist $os)
    rm -rf $dist
    let bin = ($dist | path join bin)
    mkdir $bin
    let suffix = if $os == windows { ".exe" } else { "" }
    for p in $programs { cp ($build_dir | path join apps $"($p)($suffix)") $bin }
    if $os == windows {
        stage-windows $bin $prefix
    } else {
        stage-linux $dist $prefix
        # The Linux installer ships as its own release asset, outside the payload.
        let installer_build = ($build_dir | path join installer)
        ^cmake -S v-sekai/installer -B $installer_build -G Ninja -DCMAKE_BUILD_TYPE=Release
        check "Configuring the Linux installer"
        ^cmake --build $installer_build
        check "Building the Linux installer"
        mkdir ($root | path join dist tools-linux)
        cp ($installer_build | path join ninfer-install) ($root | path join dist tools-linux)
    }
    ^git rev-parse HEAD | str trim | save -f ($dist | path join COMMIT)
    write-buildinfo $root $dist $os $arch
    ls -s ($dist | path join "**/*" | into glob) | where type == file | select name size | print
}
