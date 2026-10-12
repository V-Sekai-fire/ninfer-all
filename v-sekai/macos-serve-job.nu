# Serve models/bonsai2-27b-heretic.ninfer (text and images) on Apple silicon through mlx-vlm, as a launchd service.
# Install: nu v-sekai/macos-serve-job.nu install [--port 8090] [--host 127.0.0.1] [--draft ninfer:mtp|ninfer:dflash2]
# Remove:  nu v-sekai/macos-serve-job.nu uninstall      Log: ~/Library/Logs/ninfer-mlx-serve.log
# Backpressure: nice 5, one sequence at a time (--max-num-seqs 1). Drafting is off by default:
# on an M2 Pro both drafters decode slower than plain greedy (see v-sekai/mlx/test_drafters.py).

const label = "com.chibifire.ninfer-mlx-serve"

def plist-path [] { $env.HOME | path join Library LaunchAgents $"($label).plist" }

def "main install" [
    --model: path = "models/bonsai2-27b-heretic.ninfer"
    --host: string = "127.0.0.1"
    --port: int = 8090
    --draft: string = ""
] {
    let root = ($env.FILE_PWD | path dirname)
    let model = ($root | path join $model)
    if not ($model | path exists) { error make {msg: $"($model) is missing; run v-sekai/macos-convert-job.nu first"} }
    let log = ($env.HOME | path join Library Logs ninfer-mlx-serve.log)
    let pixi = (which pixi | get 0.path)
    let draft = if ($draft | is-empty) { [] } else { [--draft-model $draft] }
    let plist = {
        Label: $label
        ProgramArguments: [$pixi run --manifest-path ($root | path join pixi.toml) --locked -e mlx
            python ($root | path join v-sekai mlx vlm.py) serve
            --model $model --host $host --port ($port | into string) --max-num-seqs "1" ...$draft]
        WorkingDirectory: $root
        RunAtLoad: true
        KeepAlive: true
        Nice: 5
        StandardOutPath: $log
        StandardErrorPath: $log
        EnvironmentVariables: {PATH: $"($pixi | path dirname):/usr/bin:/bin:/usr/sbin:/sbin", PYTHONUNBUFFERED: "1"}
    }
    $plist | to json | save -f (plist-path)
    ^plutil -convert xml1 (plist-path)
    let uid = (^id -u | str trim)
    ^launchctl bootout $"gui/($uid)/($label)" | complete | ignore
    # bootout returns before teardown ends; bootstrapping then fails with error 5.
    for _ in 1..20 {
        if (^launchctl print $"gui/($uid)/($label)" | complete | get exit_code) != 0 { break }
        sleep 500ms
    }
    ^launchctl bootstrap $"gui/($uid)" (plist-path)
    print $"started ($label) on http://($host):($port)/v1; log ($log)"
}

def "main uninstall" [] {
    let uid = (^id -u | str trim)
    ^launchctl bootout $"gui/($uid)/($label)" | complete | ignore
    rm -f (plist-path)
    print $"removed ($label)"
}

def main [] { print "usage: nu v-sekai/macos-serve-job.nu install|uninstall" }
