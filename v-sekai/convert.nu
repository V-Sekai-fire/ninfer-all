# Convert Ternary Bonsai 2 27B Uncensored Heretic v2 into one ninfer-all v3 artifact.
# Run from the fork root after fetch-sources.nu: pixi run nu v-sekai/convert.nu
# Recipe bonsai2_27b_ternary, as WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 documents it.
# The body keeps PrismML's t2_g128_fp16 ternary codes without rounding. Vision and the
# frontend resources come from Qwen3.8-27B. MTP and DFlash2 are ProCreations' Bonsai 2 heads.

def main [
    --sources: path = "sources"
    --out: path = "models/bonsai2-27b-heretic.ninfer"
    --device: string = "cpu"
] {
    let root = ($env.FILE_PWD | path dirname)
    cd $root
    mkdir ($out | path dirname)
    (^python -m tools.convert
        --model ($sources | path join Qwen3.8-27B)
        --recipe bonsai2_27b_ternary
        --source $"ternary=($sources | path join Heretic-v2-GGUF Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-PQ2_0.gguf)"
        --source $"mtp=($sources | path join Ternary-Bonsai-2-27B-MTP model_mtp.safetensors)"
        --source $"dflash2=($sources | path join Ternary-Bonsai-2-27B-DFlash2)"
        --components text,vision,mtp,dflash2
        --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja
        --proposal
        --device $device
        --name bonsai2-27b-heretic
        --out $out)
    if $env.LAST_EXIT_CODE != 0 { error make {msg: "conversion failed"} }
    ls $out | print
}
