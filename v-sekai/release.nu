# Build and publish a dev.* Windows and Linux release of ninfer-all with Bonsai 2 Heretic v2.
# Run on Windows from the fork root, after build.nu on both platforms and convert.nu:
#   pixi run --locked -e build nu v-sekai/release.nu --tag v0.12.0-dev.1
# dist/windows comes from build.nu on this host. dist/linux and dist/tools-linux come from
# build.nu on Linux, for example in WSL2 on the same host, copied into this checkout's dist/.
#
# The release copies the desync pattern of V-Sekai-fire/service-language-model release.ps1:
# a desync chunk store whose zstd chunks are packed, unchanged, into payload-data-NNN.bin
# volumes below the GitHub asset limit. Both platforms share one store, so the model's
# chunks are stored once. payload.caidx indexes the Windows payload for setup.exe and
# desync.exe; payload-linux.caidx indexes the Linux payload for ninfer-install and desync.
# Both payloads are restored and compared file by file by SHA-256 before anything is published.

use msvc.nu load-msvc

const volume_limit = 1900000000
const desync_repo = "https://github.com/V-Sekai-fire/multiplayer-fabric-desync.git"
const desync_rev = "5140d6b273315da434a69f5bf095e7ba2427bd6e"
const repo = "V-Sekai-fire/ninfer-all"

def check [what: string] {
    if $env.LAST_EXIT_CODE != 0 { error make {msg: $"($what) failed with exit code ($env.LAST_EXIT_CODE)"} }
}

def sha-tree [dir: path] {
    ls ($dir | path join "**/*" | into glob)
    | where type == file
    | each {|f| {path: ($f.name | path relative-to $dir | str replace -a '\' '/'), sha256: (open --raw $f.name | hash sha256)} }
    | sort-by path
}

# One payload: the platform's dist tree, the artifact, the launcher, notices and VERSION.
def stage-payload [stage: path, dist: path, artifact: path, tag: string] {
    mkdir $stage
    cp -r ($dist | path join "*" | into glob) $stage
    mkdir ($stage | path join models)
    cp $artifact ($stage | path join models)
    let report = $"($artifact).conversion.json"
    if ($report | path exists) { cp $report ($stage | path join models) }
    cp v-sekai/payload/serve.nu $stage
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
        if ($pair.0 | path exists) { cp $pair.0 ($notices | path join $pair.1) }
    }
    $"($tag)\n(^git rev-parse HEAD | str trim)\n" | save -f ($stage | path join VERSION)
}

def main [
    --tag: string = "v0.12.0-dev.1"
    --artifact: path = "models/bonsai2-27b-heretic.ninfer"
    --skip-publish
] {
    if not ($tag =~ '^(v[0-9]+\.[0-9]+\.[0-9]+-)?dev\.[A-Za-z0-9.-]+$') {
        error make {msg: $"Release tag must be dev.<version> or vX.Y.Z-dev.<n>: ($tag)"}
    }
    let root = ($env.FILE_PWD | path dirname)
    cd $root
    let release_dir = ($root | path join release $tag)
    if ($release_dir | path exists) {
        error make {msg: $"Release workspace already exists: ($release_dir). Move it aside first."}
    }
    for required in [dist/windows/bin/ninfer-serve.exe dist/linux/bin/ninfer-serve dist/tools-linux/ninfer-install $artifact] {
        if not ($required | path exists) { error make {msg: $"Missing input: ($required). Run build.nu on both platforms and convert.nu."} }
    }

    let store = ($release_dir | path join desync-store)
    let assets = ($release_dir | path join assets)
    let tools = ($release_dir | path join tools)
    let desync_source = ($release_dir | path join desync-source)
    let desync = ($tools | path join desync.exe)
    let desync_linux = ($tools | path join desync)
    let installer_build = ($release_dir | path join installer-build)
    for d in [$store $assets $tools] { mkdir $d }

    load-msvc

    # setup.exe with the tag pinned into it, and the volume packer.
    ^cmake -S v-sekai/installer -B $installer_build -G Ninja -DCMAKE_BUILD_TYPE=Release $"-DNI_RELEASE_TAG=($tag)"
    check "Configuring the installer"
    ^cmake --build $installer_build
    check "Building the installer"

    # The pinned V-Sekai desync, as service-language-model builds it, for both platforms.
    ^git clone --filter=blob:none --no-checkout $desync_repo $desync_source
    check "Cloning the pinned V-Sekai desync source"
    ^git -C $desync_source checkout --detach $desync_rev
    check "Checking out the pinned V-Sekai desync revision"
    ^go build -C $desync_source -o $desync ./cmd/desync
    check "Building desync.exe"
    with-env {GOOS: linux, GOARCH: amd64, CGO_ENABLED: "0"} {
        ^go build -C $desync_source -o $desync_linux ./cmd/desync
    }
    check "Building the Linux desync"

    let payloads = [
        [os dist index];
        [windows dist/windows payload.caidx]
        [linux dist/linux payload-linux.caidx]
    ]
    for p in $payloads {
        let stage = ($release_dir | path join $"payload-($p.os)")
        stage-payload $stage $p.dist $artifact $tag
        ^$desync tar --index --store $store ($release_dir | path join $p.index) $stage
        check $"Indexing the ($p.os) payload"
    }
    ^$desync verify --store $store
    check "Desync chunk-store verification"
    ^($installer_build | path join payload-pack.exe) $store $assets
    check "Writing raw desync blob volumes"

    cp ($installer_build | path join ninfer-setup.exe) ($assets | path join setup.exe)
    cp $desync $assets
    cp $desync_linux $assets
    cp dist/tools-linux/ninfer-install $assets
    for p in $payloads { cp ($release_dir | path join $p.index) $assets }

    let names = (ls $assets | get name | path basename)
    for required in [setup.exe desync.exe payload.caidx desync ninfer-install payload-linux.caidx] {
        if $required not-in $names { error make {msg: $"Required release asset is missing: ($required)"} }
    }
    let volumes = ($names | where {|n| $n =~ '^payload-data-[0-9]{3}\.bin$' } | sort)
    if ($volumes | is-empty) { error make {msg: "The payload packer did not produce any .bin data volumes."} }
    for i in 0..<($volumes | length) {
        if ($volumes | get $i) != $"payload-data-($i | fill -a r -w 3 -c '0').bin" {
            error make {msg: "The payload .bin volume set is incomplete or out of sequence."}
        }
    }

    # Windows: restore through setup.exe exactly as a user would, offline.
    let restore_windows = ($release_dir | path join restore-windows)
    let setup = (run-external ($assets | path join setup.exe) '--from' $assets '--target' $restore_windows '--quiet' | complete)
    if $setup.exit_code != 0 { error make {msg: $"Offline restore through setup.exe failed (exit ($setup.exit_code))."} }
    # Linux: ninfer-install runs the same volume unpack and desync untar; replay it here with
    # desync.exe against payload-linux.caidx. The V-Sekai Linux install check workflow runs
    # ninfer-install online against the published release.
    let restore_linux = ($release_dir | path join restore-linux)
    ^$desync untar --index --store $store ($assets | path join payload-linux.caidx) $restore_linux
    check "Restoring the Linux payload"
    for p in [[stage restore]; [payload-windows $restore_windows] [payload-linux $restore_linux]] {
        let expected = (sha-tree ($release_dir | path join $p.stage))
        let actual = (sha-tree $p.restore)
        if $expected != $actual { error make {msg: $"($p.stage): restored paths or SHA-256 hashes differ from the staged payload."} }
        print $"Restore check ($p.stage): ($expected | length) files match by SHA-256."
    }
    # The Linux manifest lets CI prove an online ninfer-install against the published release.
    sha-tree ($release_dir | path join payload-linux) | each {|r| $"($r.sha256)  ($r.path)" } | str join "\n" | save -f ($assets | path join payload-linux.sha256)

    let oversized = (ls $assets | where size >= ($volume_limit | into filesize))
    if ($oversized | is-not-empty) { error make {msg: $"Release assets must be below ($volume_limit) bytes: ($oversized.name)"} }
    if ((ls $assets | length) > 1000) { error make {msg: "A GitHub release cannot contain more than 1000 assets."} }

    for temporary in [payload-windows payload-linux restore-windows restore-linux desync-store tools desync-source installer-build payload.caidx payload-linux.caidx] {
        rm -rf ($release_dir | path join $temporary)
    }

    let notes = ([
        "ninfer-all for one RTX 4090 (sm_89), Windows x64 and Linux x86-64, with Ternary Bonsai 2 27B Uncensored Heretic v2 as one v3 artifact: the t2_g128_fp16 ternary body, the Bonsai 2 MTP head and DFlash2 adapter, and the Qwen3.8-27B vision tower."
        ""
        "Windows: run setup.exe with an Internet connection to download this exact release, or download every asset beside setup.exe for a fully offline installation. It restores payload.caidx into C:\\ProgramData\\V-Sekai\\NInfer."
        ""
        $"Linux: chmod +x ninfer-install, then `./ninfer-install --online --tag ($tag) --target ~/ninfer`, or `--from <folder>` with desync, payload-linux.caidx and every payload-data volume. The host needs an NVIDIA driver of the CUDA 13 branch and glibc 2.28 or newer."
        ""
        "Both platforms share the payload-data volumes, which contain unchanged desync zstd chunks. Start the server from the install directory with `nu serve.nu`."
    ] | str join "\n")
    $notes | save -f ($release_dir | path join notes.md)

    if not $skip_publish {
        let files = (ls $assets | get name)
        ^gh release create $tag ...$files --repo $repo --prerelease --title $tag --notes-file ($release_dir | path join notes.md)
        check $"Publishing GitHub release ($tag)"
    }
    print $"Release assets are ready at ($assets)"
    if $skip_publish { print "Publishing skipped." }
}
