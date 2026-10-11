# Fetch the sources and convert the Bonsai 2 Heretic artifact on macOS, as one launchd oneshot.
# Install: nu v-sekai/macos-convert-job.nu install   Watch: tail -f ~/Library/Logs/ninfer-convert.log
# Credits: 4 CPU threads for torch and BLAS, 4 download workers. Backpressure: nice 10, low IO.

const label = "com.chibifire.ninfer-convert"

def plist-path [] { $env.HOME | path join Library LaunchAgents $"($label).plist" }

def "main install" [] {
    let root = ($env.FILE_PWD | path dirname)
    let log = ($env.HOME | path join Library Logs ninfer-convert.log)
    let pixi = (which pixi | get 0.path)
    let args = [$pixi run --manifest-path ($root | path join pixi.toml) --locked nu ($root | path join v-sekai macos-convert-job.nu) run]
    let plist = {
        Label: $label
        ProgramArguments: $args
        WorkingDirectory: $root
        RunAtLoad: true
        Nice: 10
        LowPriorityIO: true
        StandardOutPath: $log
        StandardErrorPath: $log
        EnvironmentVariables: {
            PATH: $"($pixi | path dirname):/usr/bin:/bin:/usr/sbin:/sbin"
            OMP_NUM_THREADS: "4"
            MKL_NUM_THREADS: "4"
            VECLIB_MAXIMUM_THREADS: "4"
            HF_HUB_DOWNLOAD_WORKERS: "4"
            PYTHONUNBUFFERED: "1"
        }
    }
    $plist | to json | save -f (plist-path)
    ^plutil -convert xml1 (plist-path)
    let uid = (^id -u | str trim)
    ^launchctl bootout $"gui/($uid)/($label)" | complete | ignore
    ^launchctl bootstrap $"gui/($uid)" (plist-path)
    print $"started ($label); log ($log)"
}

def "main run" [] {
    let root = ($env.FILE_PWD | path dirname)
    cd $root
    print $"[job] (date now | format date '%FT%T') fetch-sources start"
    nu v-sekai/fetch-sources.nu
    if $env.LAST_EXIT_CODE != 0 { error make {msg: "fetch-sources failed"} }
    print $"[job] (date now | format date '%FT%T') convert start"
    nu v-sekai/convert.nu --device cpu
    if $env.LAST_EXIT_CODE != 0 { error make {msg: "convert failed"} }
    let out = ("models/bonsai2-27b-heretic.ninfer" | path expand)
    print $"[job] (date now | format date '%FT%T') done (ls $out | get 0.size) sha256 (^shasum -a 256 $out | split row ' ' | first)"
}

def main [] { print "usage: nu v-sekai/macos-convert-job.nu install|run" }
