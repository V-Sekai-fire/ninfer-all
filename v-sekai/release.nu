# Build and publish a dev.* Linux release of ninfer-all with Bonsai 2 Heretic v2.
# Run on Linux (Fedora 44, native or WSL2) from the fork root, after build.nu and convert.nu:
#   pixi run --locked -e build nu v-sekai/release.nu --tag v0.12.0-dev.1
#
# The release copies the desync pattern of V-Sekai-fire/service-language-model release.ps1:
# a desync chunk store whose zstd chunks are packed, unchanged, into payload-data-NNN.bin
# volumes below the GitHub asset limit. payload-linux.caidx indexes the payload for
# ninfer-install and desync. The payload is restored and compared file by file by SHA-256
# before anything is published. The release is uploaded into an existing release of the same
# tag, or created when there is none.

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

# The payload: the dist tree, the artifact, the launcher, notices and VERSION.
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
    if $nu.os-info.name != linux { error make {msg: "release.nu runs only on Linux (Fedora 44, native or WSL2)."} }
    for required in [dist/linux/bin/ninfer-serve dist/tools-linux/ninfer-install $artifact] {
        if not ($required | path exists) { error make {msg: $"Missing input: ($required). Run build.nu and convert.nu."} }
    }

    let store = ($release_dir | path join desync-store)
    let assets = ($release_dir | path join assets)
    let tools = ($release_dir | path join tools)
    let desync_source = ($release_dir | path join desync-source)
    let desync = ($tools | path join desync)
    let installer_build = ($release_dir | path join installer-build)
    for d in [$store $assets $tools] { mkdir $d }

    # The volume packer.
    ^cmake -S v-sekai/installer -B $installer_build -G Ninja -DCMAKE_BUILD_TYPE=Release
    check "Configuring the installer"
    ^cmake --build $installer_build
    check "Building the installer"

    # The pinned V-Sekai desync, as service-language-model builds it.
    ^git clone --filter=blob:none --no-checkout $desync_repo $desync_source
    check "Cloning the pinned V-Sekai desync source"
    ^git -C $desync_source checkout --detach $desync_rev
    check "Checking out the pinned V-Sekai desync revision"
    with-env {GOOS: linux, GOARCH: amd64, CGO_ENABLED: "0"} {
        ^go build -C $desync_source -o $desync ./cmd/desync
    }
    check "Building desync"

    let stage = ($release_dir | path join payload-linux)
    let index = ($release_dir | path join payload-linux.caidx)
    stage-payload $stage dist/linux $artifact $tag
    ^$desync tar --index --store $store $index $stage
    check "Indexing the payload"
    ^$desync verify --store $store
    check "Desync chunk-store verification"
    ^($installer_build | path join payload-pack) $store $assets
    check "Writing raw desync blob volumes"

    cp $desync $assets
    cp dist/tools-linux/ninfer-install $assets
    cp $index $assets

    let names = (ls $assets | get name | path basename)
    for required in [desync ninfer-install payload-linux.caidx] {
        if $required not-in $names { error make {msg: $"Required release asset is missing: ($required)"} }
    }
    let volumes = ($names | where {|n| $n =~ '^payload-data-[0-9]{3}\.bin$' } | sort)
    if ($volumes | is-empty) { error make {msg: "The payload packer did not produce any .bin data volumes."} }
    for i in 0..<($volumes | length) {
        if ($volumes | get $i) != $"payload-data-($i | fill -a r -w 3 -c '0').bin" {
            error make {msg: "The payload .bin volume set is incomplete or out of sequence."}
        }
    }

    # ninfer-install runs the same volume unpack and desync untar; replay it here against
    # payload-linux.caidx. The V-Sekai Linux install check workflow runs ninfer-install online
    # against the published release.
    let restore = ($release_dir | path join restore-linux)
    ^$desync untar --index --store $store ($assets | path join payload-linux.caidx) $restore
    check "Restoring the payload"
    let expected = (sha-tree $stage)
    let actual = (sha-tree $restore)
    if $expected != $actual { error make {msg: "payload-linux: restored paths or SHA-256 hashes differ from the staged payload."} }
    print $"Restore check payload-linux: ($expected | length) files match by SHA-256."
    # The manifest lets CI prove an online ninfer-install against the published release.
    sha-tree $stage | each {|r| $"($r.sha256)  ($r.path)" } | str join "\n" | save -f ($assets | path join payload-linux.sha256)

    let oversized = (ls $assets | where size >= ($volume_limit | into filesize))
    if ($oversized | is-not-empty) { error make {msg: $"Release assets must be below ($volume_limit) bytes: ($oversized.name)"} }
    if ((ls $assets | length) > 1000) { error make {msg: "A GitHub release cannot contain more than 1000 assets."} }

    for temporary in [payload-linux restore-linux desync-store tools desync-source installer-build payload-linux.caidx] {
        rm -rf ($release_dir | path join $temporary)
    }

    ([
        "ninfer-all for one RTX 4090 (sm_89) on Linux x86-64 (Fedora 44, native or WSL2), with Ternary Bonsai 2 27B Uncensored Heretic v2 as one v3 artifact: the t2_g128_fp16 ternary body, the Bonsai 2 MTP head and DFlash2 adapter, and the Qwen3.8-27B vision tower."
        ""
        $"chmod +x ninfer-install, then `./ninfer-install --online --tag ($tag) --target ~/ninfer`, or `--from <folder>` with desync, payload-linux.caidx and every payload-data volume. The host needs an NVIDIA driver of the CUDA 13 branch and glibc 2.28 or newer. Start the server from the install directory with `nu serve.nu`."
    ] | str join "\n") | save -f ($release_dir | path join notes.md)

    if not $skip_publish {
        let files = (ls $assets | get name)
        let exists = ((do { ^gh release view $tag --repo $repo --json tagName } | complete).exit_code == 0)
        if $exists {
            ^gh release upload $tag ...$files --repo $repo
            check $"Uploading the assets to ($tag)"
            let body = (^gh release view $tag --repo $repo --json body --jq .body)
            let notes = ($release_dir | path join notes-combined.md)
            $"($body)\n\n(open --raw ($release_dir | path join notes.md))\n" | save -f $notes
            ^gh release edit $tag --repo $repo --notes-file $notes
            check $"Adding the notes to ($tag)"
        } else {
            ^gh release create $tag ...$files --repo $repo --prerelease --title $tag --notes-file ($release_dir | path join notes.md)
            check $"Publishing GitHub release ($tag)"
        }
    }
    print $"Release assets are ready at ($assets)"
    if $skip_publish { print "Publishing skipped." }
}
