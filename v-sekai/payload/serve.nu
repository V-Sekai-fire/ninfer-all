# Serve Ternary Bonsai 2 27B Uncensored Heretic v2 with ninfer-all on one RTX 4090, on Windows or Linux.
# Run from the install directory: nu serve.nu
# Speculation: the Bonsai-trained MTP head, three drafts, scored on the proposal head.
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
    (^($root | path join bin $exe) ($root | path join models bonsai2-27b-heretic.ninfer)
        --host $host --port $port
        --model-id bonsai2-27b-heretic
        --max-context $max_context --kv-capacity auto --kv-dtype $kv_dtype
        --max-concurrency $max_concurrency
        --gdn-state-fp16
        --spec mtp --draft-tokens 3 --lm-head-draft
        --vision --vision-residency overlay --vision-max-merged 12288)
}
