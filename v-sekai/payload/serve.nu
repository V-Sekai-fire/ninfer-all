# Serve Ternary Bonsai 2 27B Uncensored Heretic v2 with ninfer-all on one RTX 4090, on Windows or Linux.
# Run from the install directory: nu serve.nu
# Speculation: DFlash2 with six drafts, the fastest single-stream decode measured on an RTX 4090 at 275 W.
# Routes: the first start calibrates this GPU and saves device-profile-<device>.json beside this
# script (about a minute); later starts load it. Delete the file after changing GPU or power limit.
# Vision: the tower waits in pinned host memory and borrows device memory per encode.

def main [
    --host: string = "127.0.0.1"
    --port: int = 8080
    --max-context: int = 262144
    --kv-dtype: string = "rk4v4-e8"
    --max-concurrency: int = 3
    --device: int = 0
] {
    let root = $env.FILE_PWD
    $env.CUDA_VISIBLE_DEVICES = ($device | into string)
    let exe = if $nu.os-info.name == windows { 'ninfer-serve.exe' } else { 'ninfer-serve' }
    let profile = ($root | path join $"device-profile-($device).json")
    let routes = if ($profile | path exists) { 'auto' } else { 'calibrate' }
    (^($root | path join bin $exe) ($root | path join models bonsai2-27b-heretic.ninfer)
        --host $host --port $port
        --model-id bonsai2-27b-heretic
        --max-context $max_context --kv-capacity auto --kv-dtype $kv_dtype
        --max-concurrency $max_concurrency
        --gdn-state-fp16
        --spec dflash2 --draft-tokens 6
        --device-profile $routes --device-profile-path $profile
        --vision --vision-residency overlay --vision-max-merged 12288)
}
