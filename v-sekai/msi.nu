# Package the Windows build and the split Bonsai 2 Heretic v2 artifact as one MSI with external CABs.
# Run on Windows from the fork root, inside the build environment, after build.nu and convert.nu:
#   pixi run --locked -e build nu v-sekai/msi.nu --tag v0.12.0-dev.1
# Inputs:
#   dist/windows/                          build.nu output (ninfer-serve.exe, its DLLs, BUILDINFO.json)
#   models/bonsai2-27b-heretic.ninfer*     convert.nu --max-file-bytes output: the entry and its .part-NNNN files
#   sources/**/LICENSE, NOTICE, README.md  the notices of the conversion sources (optional)
# Output, in release/<tag>-msi/:
#   ninfer-all-<tag>-windows-x64.msi, ninfer1.cab ... ninferN.cab, payload-msi.sha256, SHA256SUMS
# WiX 5 (`wix` on PATH, from `dotnet tool install --global wix --version 5.0.2`) builds the MSI.
#
#   nu v-sekai/msi.nu verify --target 'C:\Program Files\V-Sekai\NInfer' --manifest <payload-msi.sha256>
# checks an installed tree against the manifest, file by file, by SHA-256.

use msvc.nu load-msvc

# A GitHub release asset must be below 2 GiB, and so must a CAB. The converter splits the model
# into files of at most this many bytes, and the MSI puts each one in a CAB of its own.
const max_asset_bytes = 1900000000
const model = "bonsai2-27b-heretic.ninfer"

def check [what: string] {
    if $env.LAST_EXIT_CODE != 0 { error make {msg: $"($what) failed with exit code ($env.LAST_EXIT_CODE)"} }
}

# nu cannot parse a glob pattern that starts with a Windows path, so glob from inside the folder.
# Returns absolute paths.
def files-under [dir: path, pattern: string] {
    do { cd $dir; glob $pattern --no-dir }
}

def sha-tree [dir: path] {
    let dir = ($dir | path expand)
    files-under $dir "**/*"
    | each {|f| {path: ($f | path expand | path relative-to $dir | str replace -a '\' '/'), sha256: (open --raw $f | hash sha256)} }
    | sort-by path
}

# vX.Y.Z[-dev.N|-beta.N|-rc.N] -> X.Y.Z.B. MSI compares only X.Y.Z; B tells builds apart.
def msi-version [tag: string] {
    let m = ($tag | parse -r '^v(?<x>\d+)\.(?<y>\d+)\.(?<z>\d+)(?:-(?<stage>dev|beta|rc)\.(?<n>\d+))?$')
    if ($m | is-empty) { error make {msg: $"Release tag must be vX.Y.Z or vX.Y.Z-dev|beta|rc.N: ($tag)"} }
    let m = ($m | first)
    let base = match $m.stage { "dev" => 0, "beta" => 1000, "rc" => 2000, _ => 3000 }
    let n = if ($m.n | is-empty) { 0 } else { $m.n | into int }
    if ($m.x | into int) > 255 or ($m.y | into int) > 255 or ($m.z | into int) > 65535 or $n >= 1000 {
        error make {msg: $"($tag) does not fit an MSI ProductVersion"}
    }
    $"($m.x).($m.y).($m.z).($base + $n)"
}

def stage-notices [stage: path] {
    let notices = ($stage | path join licenses)
    mkdir $notices
    cp LICENSE ($notices | path join ninfer-all-LICENSE)
    for pair in [
        [sources/Qwen3.8-27B/LICENSE Qwen3.8-27B-LICENSE]
        [sources/Ternary-Bonsai-2-27B-MTP/LICENSE Ternary-Bonsai-2-27B-MTP-LICENSE]
        [sources/Ternary-Bonsai-2-27B-MTP/NOTICE Ternary-Bonsai-2-27B-MTP-NOTICE]
        [sources/Ternary-Bonsai-2-27B-DFlash2/LICENSE Ternary-Bonsai-2-27B-DFlash2-LICENSE]
        [sources/Ternary-Bonsai-2-27B-DFlash2/NOTICE Ternary-Bonsai-2-27B-DFlash2-NOTICE]
        [sources/Heretic-v2-GGUF/README.md Heretic-v2-GGUF-README.md]
    ] {
        if ($pair.0 | path exists) { cp $pair.0 ($notices | path join $pair.1) } else { print $"Notice not found: ($pair.0)" }
    }
}

def main [
    --tag: string = "v0.12.0-dev.1"
    --dist: path = "dist/windows"
    --models: path = "models"
    --move-model    # move the model files into the stage instead of copying them (saves 9.5 GB on CI)
] {
    let version = (msi-version $tag)
    let root = ($env.FILE_PWD | path dirname)
    cd $root
    let out = ($root | path join release $"($tag)-msi")
    if ($out | path exists) { error make {msg: $"Output folder already exists: ($out). Move it aside first."} }
    let stage = ($root | path join release $"($tag)-msi-stage")
    rm -rf $stage
    for required in [($dist | path join bin ninfer-serve.exe) ($dist | path join BUILDINFO.json) ($models | path join $model)] {
        if not ($required | path exists) { error make {msg: $"Missing input: ($required). Run build.nu and convert.nu first."} }
    }
    let model_files = (files-under $models $"($model)*"
        | where {|f| ($f | path basename) =~ '\.ninfer(\.part-\d{4,})?$' or ($f | path basename) == $"($model).conversion.json" }
        | sort)
    let oversized = ($model_files | where {|f| (ls $f | first | get size) >= ($max_asset_bytes | into filesize) })
    if ($oversized | is-not-empty) {
        error make {msg: $"Model files must be below ($max_asset_bytes) bytes; convert with --max-file-bytes: ($oversized)"}
    }

    load-msvc

    # The service host, built with the pinned toolchain beside setup.exe's sources.
    let host_build = ($root | path join release $"($tag)-msi-host")
    ^cmake --fresh -S v-sekai/installer -B $host_build -G Ninja -DCMAKE_BUILD_TYPE=Release
    check "Configuring the service host"
    ^cmake --build $host_build --target ninfer-service
    check "Building the service host"

    # Stage: bin/ (the build plus the MSVC runtime DLLs), models/, licenses/, the flags file
    # and the service host.
    mkdir ($stage | path join bin) ($stage | path join models) ($stage | path join service)
    for f in (do { cd ($dist | path join bin); glob "*" }) { cp -r $f ($stage | path join bin) }
    # Batteries included: the app-local MSVC runtime, so no redistributable install is needed.
    let crt = (files-under ($env.VCToolsRedistDir | path join x64) "Microsoft.VC*.CRT/*.dll")
    if ($crt | is-empty) { error make {msg: $"No MSVC runtime DLLs under ($env.VCToolsRedistDir)"} }
    for dll in $crt {
        let target = ($stage | path join bin ($dll | path basename))
        if not ($target | path exists) { cp $dll $target }
    }
    for f in $model_files {
        if $move_model { mv $f ($stage | path join models) } else { cp $f ($stage | path join models) }
    }
    stage-notices $stage
    cp ($dist | path join BUILDINFO.json) $stage
    $"($tag)\n(^git rev-parse HEAD | str trim)\n" | save -f ($stage | path join VERSION)
    cp v-sekai/msi/ninfer-serve.args $stage
    cp ($host_build | path join ninfer-service.exe) ($stage | path join service)

    # The installed tree, as the manifest the install check compares against. service/ installs
    # into bin/, and the flags file into %ProgramData%, which the manifest leaves out.
    let manifest = (sha-tree $stage
        | where path != ninfer-serve.args
        | each {|r| if ($r.path | str starts-with service/) { $r | update path ($r.path | str replace service/ bin/) } else { $r } }
        | sort-by path)

    mkdir $out
    let msi = ($out | path join $"ninfer-all-($tag)-windows-x64.msi")
    ^wix build -arch x64 -d $"Version=($version)" -d $"Stage=($stage)" -o $msi v-sekai/msi/ninfer.wxs
    check "Building the MSI"
    # wix writes a .wixpdb beside the MSI; it is not a release asset.
    for f in (files-under $out "*.wixpdb") { rm -f $f }
    $manifest | each {|r| $"($r.sha256)  ($r.path)" } | str join "\n" | save -f ($out | path join payload-msi.sha256)

    let assets = (ls $out | where type == file)
    let too_big = ($assets | where size >= ($max_asset_bytes | into filesize))
    if ($too_big | is-not-empty) { error make {msg: $"Release assets must be below ($max_asset_bytes) bytes: ($too_big.name)"} }
    let cabs = ($assets | where {|a| ($a.name | path basename) =~ '\.cab$' })
    if ($cabs | is-empty) { error make {msg: "wix wrote no external CABs"} }
    $assets | each {|a| $"(open --raw $a.name | hash sha256)  ($a.name | path basename)" } | str join "\n" | save -f ($out | path join SHA256SUMS)

    rm -rf $stage $host_build
    print $"MSI ($version): ($msi)"
    ls $out | select name size | print
}

# Compare an installed tree with payload-msi.sha256: every listed file must exist with its hash.
def "main verify" [--target: path, --manifest: path] {
    let expected = (open --raw $manifest | lines | where {|l| $l | is-not-empty } | parse '{sha256}  {path}' | sort-by path)
    let actual = (sha-tree $target | where {|r| $r.path in $expected.path })
    let missing = ($expected.path | where {|p| $p not-in $actual.path })
    if ($missing | is-not-empty) { error make {msg: $"Missing after install: ($missing)"} }
    let differ = ($expected | zip $actual | where {|p| $p.0.sha256 != $p.1.sha256 } | each {|p| $p.0.path })
    if ($differ | is-not-empty) { error make {msg: $"SHA-256 differs after install: ($differ)"} }
    print $"Install check: ($expected | length) files match by SHA-256."
}
