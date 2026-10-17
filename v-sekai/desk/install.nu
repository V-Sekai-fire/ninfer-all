# Install or refresh one Bonsai 2 Heretic server on the Windows desk (RTX 4090 or RTX 3090).
# Runs on the desk. Every step is idempotent, so running it again repairs or updates the card.
#   nu install.nu 4090 --dist C:/ninfer-scratch/dist-windows.zip
#   nu install.nu 3090 --url-file C:/ninfer-scratch/u.txt   (deploy.nu on the Mac writes the file)
# --url-file holds a short-lived GitHub artifact download URL; the file is deleted after use.
# Without --dist or --url-file the installed build is kept and only the launcher, tasks and
# tunnel are rewritten.
#
# Layout:
#   C:/ninfer-<card>/                         engine build (bin/ninfer-serve.exe, SHA256SUMS, BUILDINFO.json)
#   C:/ninfer-all-models/<model>              the published artifact, checked by sha256
#   %LOCALAPPDATA%/V-Sekai/llm/               api-key.txt, serve-ninfer-<card>.cmd, logs/
#   ~/.ninfer-stunnel/stunnel.conf            one mTLS section per installed card
#   tasks \V-Sekai\llm-ninfer-<card>, \V-Sekai\ninfer-stunnel (logon, highest)

const cards = {
    "4090": {arch: "89", gpu: "RTX 4090", port: 8091, tunnel: 8443}
    "3090": {arch: "86", gpu: "RTX 3090", port: 8092, tunnel: 8444}
}
const model_name = "bonsai2-27b-heretic.ninfer"
const model_sha256 = "36b73f569720ad3756f46a761b7ea8fdf6ad85b3e9721617d3c4739d42905765"
const model_url = "https://huggingface.co/datasets/chibifire/ninfer-bonsai2-heretic/resolve/main/models/next/bonsai2-27b-heretic.ninfer"
const models_dir = "C:/ninfer-all-models"
const stunnel_exe = 'C:\Program Files (x86)\stunnel\bin\stunnel.exe'
# The server flags measured on the desk 4090 (weftspun RFD 2311).
const serve_flags = "--max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --vision --max-concurrency 1 --spec dflash2 --draft-tokens 5 --no-thinking"

def llm-dir [] { $env.LOCALAPPDATA | path join V-Sekai llm }
def card-dir [card: string] { $"C:/ninfer-($card)" }
def win [p: string] { $p | str replace -a '/' '\' }

# Machine identity from the Bao agent cert directory: ~/.bao-creds/<name>/ holds srv-key.pem.
def identity [] {
    let dirs = (glob ($env.USERPROFILE | path join .bao-creds * srv-key.pem | str replace -a '\' '/'))
    if ($dirs | is-empty) { error make {msg: "no ~/.bao-creds/<id>/srv-key.pem; enroll the desk with Bao first"} }
    $dirs | first | path dirname
}

def install-dist [card: string, zip: path] {
    let c = ($cards | get $card)
    let dir = (card-dir $card)
    let staging = $"($dir).new"
    if ($staging | path exists) { rm -rf $staging }
    mkdir $staging
    ^tar -xf $zip -C $staging
    if $env.LAST_EXIT_CODE != 0 { error make {msg: $"tar failed on ($zip)"} }
    let arch = (open ($staging | path join BUILDINFO.json) | get cuda_arch | into string)
    if $arch != $c.arch { error make {msg: $"($zip) is built for sm_($arch); the ($card) needs sm_($c.arch)"} }
    let bad = (do { cd $staging; open --raw SHA256SUMS | lines | where ($it | str trim) != "" | each {|l|
        let p = ($l | split row -n 2 " ")
        let f = ($p.1 | str trim | str trim -l -c '*')
        if (open --raw $f | hash sha256) != $p.0 { $f }
    } | compact })
    if ($bad | is-not-empty) { error make {msg: $"checksum mismatch in ($zip): ($bad)"} }
    stop-server $card
    if ($dir | path exists) { rm -rf $dir }
    mv $staging $dir
    print $"($card): installed sm_($arch) build (open --raw ($dir | path join COMMIT) | str trim) into ($dir)"
}

def ensure-model [] {
    let model = ($models_dir | path join $model_name)
    let stamp = $"($model).sha256ok"
    mkdir $models_dir
    let fresh = ($model | path exists) and ($stamp | path exists) and (open --raw $stamp | str trim) == (stamp-of $model)
    if $fresh { return }
    if not ($model | path exists) {
        print $"downloading ($model_name)"
        ^curl.exe -L --retry 5 -C - -o $model $model_url
    }
    let sum = (^certutil -hashfile (win $model) SHA256 | lines | get 1 | str replace -a ' ' '' | str lowercase)
    if $sum != $model_sha256 { error make {msg: $"($model) has sha256 ($sum), expected ($model_sha256); delete it and run again"} }
    stamp-of $model | save --force --raw $stamp
    print $"model sha256 ok: ($model_sha256)"
}
def stamp-of [f: path] { let s = (ls $f | first); $"($model_sha256) ($s.size | into int) ($s.modified | format date '%s')" }

def ensure-key [] {
    let kf = (llm-dir | path join api-key.txt)
    if not ($kf | path exists) { random chars --length 43 | save --raw $kf }
    $kf
}

# The CUDA index of the card in PCI bus order (the order nvidia-smi prints).
def gpu-index [card: string] {
    let gpu = ($cards | get $card | get gpu)
    let rows = (^nvidia-smi --query-gpu=index,name --format=csv,noheader | lines | parse "{index}, {name}" | where name =~ $gpu)
    if ($rows | length) != 1 { error make {msg: $"expected one ($gpu), found ($rows | length)"} }
    $rows.0.index
}

def write-launcher [card: string] {
    let c = ($cards | get $card)
    let d = (llm-dir)
    mkdir ($d | path join logs)
    let kf = (ensure-key)
    let exe = (win ((card-dir $card) | path join bin ninfer-serve.exe))
    let model = (win ($models_dir | path join $model_name))
    let log = $"(win ($d | path join logs))\\ninfer-($card).log"
    # The loop restarts the server 10 s after it exits for any reason (crash, driver reset, kill).
    let cmd = ([
        "@echo off"
        "rem Written by v-sekai/desk/install.nu; edit that script, not this file."
        "set CUDA_DEVICE_ORDER=PCI_BUS_ID"
        ":serve"
        $"set /p KEY=<\"(win $kf)\""
        $"\"($exe)\" \"($model)\" --device (gpu-index $card) --host 127.0.0.1 --port ($c.port) --api-key %KEY% --model-id bonsai-27b ($serve_flags) >> \"($log)\" 2>&1"
        $"echo %date% %time% launcher: ninfer-serve exited with %errorlevel%, restarting in 10 s >> \"($log)\""
        "timeout /t 10 /nobreak >nul"
        "goto serve"
    ] | str join "\r\n") + "\r\n"
    let f = ($d | path join $"serve-ninfer-($card).cmd")
    $cmd | save --force --raw $f
    let tr = $"conhost.exe --headless cmd /c \"(win $f)\""
    ^schtasks /create /f /tn $"\\V-Sekai\\llm-ninfer-($card)" /sc onlogon /ru $env.USERNAME /rl highest /tr $tr | ignore
}

def kill-matching [image: string, pattern: string] {
    ^powershell -NoProfile -Command $"Get-CimInstance Win32_Process -Filter \"name='($image)'\" | Where-Object CommandLine -match '($pattern)' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
}

def stop-server [card: string] {
    ^schtasks /end /tn $"\\V-Sekai\\llm-ninfer-($card)" o+e>| ignore
    # Stop the launcher's restart loop first, then the server it started.
    kill-matching cmd.exe $"serve-ninfer-($card)\\.cmd"
    kill-matching ninfer-serve.exe $"--port ($cards | get $card | get port) "
}

# Start the card's task and wait for its log to say it is listening, or fail with the log tail.
def start-server [card: string] {
    let log = (llm-dir | path join logs $"ninfer-($card).log")
    # The launcher appends to the log, and the old server's cmd may still hold it open, so read
    # only what this start writes after the current end of the file.
    let offset = (if ($log | path exists) { ls $log | get 0.size | into int } else { 0 })
    let fresh = {|| if ($log | path exists) { open --raw $log | into binary | bytes at $offset.. | decode utf-8 } else { "" } }
    ^schtasks /run /tn $"\\V-Sekai\\llm-ninfer-($card)" | ignore
    mut seen = false
    for _ in 1..180 {
        sleep 1sec
        let text = (do $fresh)
        if ($text | str contains "listening on") { $seen = true; break }
        if ($text =~ "(?i)fatal|\\) failed:") { break }
    }
    let tail = (do $fresh | lines | last 12 | str join "\n")
    if not $seen { error make {msg: $"($card) server did not start:\n($tail)"} }
    let kf = (ensure-key)
    let code = (^curl.exe -s -o NUL -w "%{http_code}" -H $"Authorization: Bearer (open --raw $kf | str trim)" $"http://127.0.0.1:($cards | get $card | get port)/v1/models")
    if $code != "200" { error make {msg: $"($card) answers /v1/models with ($code)"} }
    print $"($card): serving on 127.0.0.1:($cards | get $card | get port), /v1/models 200"
}

# One stunnel section per installed card, with the desk's Bao serving cert.
def write-tunnel [] {
    let id = (identity)
    let fullchain = ($env.USERPROFILE | path join .cua-tunnel srv-fullchain.pem)
    let ca = ($env.USERPROFILE | path join .cua-tunnel client-ca.pem)
    for f in [$fullchain $ca] { if not ($f | path exists) { error make {msg: $"($f) is missing"} } }
    let ip = (^tailscale ip -4 | str trim)
    let installed = ($cards | transpose card c | where {|r| (card-dir $r.card | path join bin ninfer-serve.exe) | path exists })
    let d = ($env.USERPROFILE | path join .ninfer-stunnel)
    mkdir $d
    let head = [
        "; Written by v-sekai/desk/install.nu. Bonsai 2 Heretic over mTLS for the Mac (mac-mini-fire-01b8f5)."
        $"output = (win ($d | path join stunnel.log))"
        "debug = 5"
    ]
    let sections = ($installed | each {|r| [
        ""
        $"[ninfer-($r.card)]"
        $"accept = ($ip):($r.c.tunnel)"
        $"connect = 127.0.0.1:($r.c.port)"
        $"cert = (win $fullchain)"
        $"key = (win ($id | path join srv-key.pem))"
        $"CAfile = (win $ca)"
        "verifyChain = yes"
        "requireCert = yes"
        "checkHost = mac-mini-fire-01b8f5.agents.weftspun"
        "sslVersionMin = TLSv1.3"
    ] } | flatten)
    (($head ++ $sections) | str join "\r\n") + "\r\n" | save --force --raw ($d | path join stunnel.conf)
    let ports = ($installed | get c.tunnel | str join ",")
    ^netsh advfirewall firewall delete rule name=ninfer-stunnel o+e>| ignore
    ^netsh advfirewall firewall add rule name=ninfer-stunnel dir=in action=allow protocol=TCP $"localport=($ports)" remoteip=100.64.0.0/10 $"program=($stunnel_exe)" | ignore
    ^schtasks /create /f /tn "\\V-Sekai\\ninfer-stunnel" /sc onlogon /ru $env.USERNAME /rl highest /tr $"\"($stunnel_exe)\" -quiet \"(win ($d | path join stunnel.conf))\"" | ignore
    # Restart only this tunnel; other stunnel instances (cua-tunnel) keep running.
    ^schtasks /end /tn "\\V-Sekai\\ninfer-stunnel" o+e>| ignore
    kill-matching stunnel.exe "ninfer-stunnel"
    ^schtasks /run /tn "\\V-Sekai\\ninfer-stunnel" | ignore
    print $"tunnel: ($installed | get card | str join ', ') on ($ip) ports ($ports)"
}

def main [card: string, --dist: path, --url-file: path] {
    if $card not-in ($cards | columns) { error make {msg: $"card must be one of ($cards | columns | str join ', ')"} }
    ensure-model
    let dist = if $url_file != null {
        let zip = $"C:/ninfer-scratch/dist-windows-sm($cards | get $card | get arch).zip"
        mkdir ($zip | path dirname)
        let url = (open --raw $url_file | str trim)
        rm $url_file
        ^curl.exe -sSL --retry 3 -o $zip $url
        if $env.LAST_EXIT_CODE != 0 { error make {msg: "artifact download failed"} }
        $zip
    } else { $dist }
    if $dist != null { install-dist $card $dist }
    if not ((card-dir $card) | path join bin ninfer-serve.exe | path exists) {
        error make {msg: $"no build in (card-dir $card); pass --dist <dist-windows zip>"}
    }
    write-launcher $card
    stop-server $card
    start-server $card
    write-tunnel
}
