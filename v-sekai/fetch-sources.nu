# Download every conversion source for the Bonsai 2 Heretic v2 artifact.
# Run from the fork root: pixi run --locked nu v-sekai/fetch-sources.nu
# The Qwen3.8-27B checkpoint gives config, tokenizer, frontend resources and
# the vision tower. Every vision tensor is in shard 1, so only that shard is fetched.

def main [--sources: path = "sources"] {
    mkdir $sources
    let qwen = ($sources | path join "Qwen3.8-27B")
    ^hf download Qwen/Qwen3.8-27B chat_template.jinja config.json generation_config.json merges.txt model.safetensors.index.json preprocessor_config.json tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json LICENSE model-00001-of-00018.safetensors --local-dir $qwen
    ^hf download ProCreations/Ternary-Bonsai-2-27B-MTP model_mtp.safetensors mtp_config.json LICENSE NOTICE --local-dir ($sources | path join "Ternary-Bonsai-2-27B-MTP")
    ^hf download ProCreations/Ternary-Bonsai-2-27B-DFlash2 model.safetensors config.json LICENSE NOTICE --local-dir ($sources | path join "Ternary-Bonsai-2-27B-DFlash2")
    ^hf download OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-GGUF Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-PQ2_0.gguf README.md --local-dir ($sources | path join "Heretic-v2-GGUF")
    ls -s $sources | print
}
