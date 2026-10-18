# One OpenAI-compatible port on the Mac for every Bonsai 2 Heretic server (LiteLLM proxy).
#   nu v-sekai/router/router.nu install   copy to ~/.local/share/model-router, pixi install
#                                              --locked, write the LaunchAgent, set the Hermes provider
#   nu v-sekai/router/router.nu run       what the LaunchAgent runs (127.0.0.1:18000)
#   nu v-sekai/router/router.nu check     witnesses: no key 401, models list, one chat
# Backends and model names are in config.yaml. The router reads only the keys it needs from
# ~/.hermes/.env and creates MODEL_ROUTER_API_KEY there on first install.

const label = "com.chibifire.model-router"
const port = 18000
const keys = [BONSAI_DESK_API_KEY BONSAI_FEDORA_API_KEY MODEL_ROUTER_API_KEY]

def home-dir [] { $env.HOME | path join .local share model-router }
def env-file [] { $env.HOME | path join .hermes .env }
def plist-path [] { $env.HOME | path join Library LaunchAgents $"($label).plist" }

def env-value [name: string] {
    let row = (open --raw (env-file) | lines | where $it starts-with $"($name)=")
    if ($row | is-empty) { error make {msg: $"($name) is not in ~/.hermes/.env"} }
    $row | first | str substring (($name | str length) + 1)..
}

def "main install" [] {
    let dst = (home-dir)
    mkdir $dst
    for f in [pixi.toml pixi.lock config.yaml router.nu] { cp -f ($env.FILE_PWD | path join $f) ($dst | path join $f) }
    ^pixi install --locked --manifest-path ($dst | path join pixi.toml)
    if (open --raw (env-file) | lines | where $it starts-with "MODEL_ROUTER_API_KEY=" | is-empty) {
        $"\nMODEL_ROUTER_API_KEY=sk-(random chars --length 40)\n" | save --append --raw (env-file)
    }
    let log = ($env.HOME | path join Library Logs model-router.log)
    let nu_bin = (which nu | get 0.path)
    $'<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>($label)</string>
  <key>ProgramArguments</key><array><string>($nu_bin)</string><string>($dst | path join router.nu)</string><string>run</string></array>
  <key>WorkingDirectory</key><string>($dst)</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>($env.HOME)/.pixi/bin:/opt/homebrew/bin:/usr/bin:/bin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>($log)</string>
  <key>StandardErrorPath</key><string>($log)</string>
</dict></plist>
' | save --force (plist-path)
    ^plutil -lint (plist-path)
    let p = "providers.bonsai"
    for kv in [
        [name "Bonsai 2 Heretic (model router)"] [base_url $"http://127.0.0.1:($port)/v1"]
        [key_env MODEL_ROUTER_API_KEY] [model bonsai] [default_model bonsai]
        [api_mode chat_completions] [context_length "262144"] [discover_models "true"]
    ] { ^hermes config set $"($p).($kv.0)" $kv.1 | ignore }
    print $"installed into ($dst); Hermes provider `bonsai` -> 127.0.0.1:($port)"
    print "Load it from a terminal (the Hermes gateway refuses launchctl bootstrap):"
    print $"  launchctl bootout gui/$\(id -u\)/($label) 2>/dev/null; launchctl bootstrap gui/$\(id -u\) (plist-path)"
}

def "main run" [] {
    cd (home-dir)
    let envs = ($keys | each {|k| {($k): (env-value $k)} } | into record)
    with-env ($envs | merge {LITELLM_TELEMETRY: "False", DISABLE_ADMIN_UI: "True"}) {
        exec pixi run --locked --manifest-path ./pixi.toml litellm --config ./config.yaml --host 127.0.0.1 --port ($port | into string) --num_workers 1
    }
}

def "main check" [] {
    let base = $"http://127.0.0.1:($port)"
    let key = (env-value MODEL_ROUTER_API_KEY)
    let no_key = (^curl -s -o /dev/null -w "%{http_code}" $"($base)/v1/models")
    let models = (^curl -s -H $"Authorization: Bearer ($key)" $"($base)/v1/models" | from json | get data.id)
    let body = {model: bonsai, temperature: 0, max_tokens: 16, messages: [{role: user, content: "Reply with exactly: router ok"}]}
    let r = (^curl -s -D - -H $"Authorization: Bearer ($key)" -H "Content-Type: application/json" $"($base)/v1/chat/completions" -d ($body | to json))
    let backend = ($r | lines | where ($it | str lowercase | str starts-with "x-litellm-model-id:") | first | split row ":" | last | str trim)
    let reply = ($r | lines | last | from json | get choices.0.message.content)
    print {no_key: $no_key, models: ($models | str join " "), bonsai_backend: $backend, reply: $reply}
    if $no_key != "401" { error make {msg: "router accepted a request without a key"} }
}

def main [] { help main }
