# Deploy one Bonsai 2 Heretic card on the Windows desk from the Mac, then test it.
#   nu v-sekai/desk/deploy.nu 3090             newest unexpired dist-windows-sm86 artifact
#   nu v-sekai/desk/deploy.nu 4090 --run <id>  that run's sm_89 artifact
#   nu v-sekai/desk/deploy.nu 4090 --keep-build  rewrite launcher/tasks/tunnel only
# Builds come from the `V-Sekai Windows build` workflow (input arch 86 or 89); the sm_89 build
# of the MSI workflow (artifact dist-windows) also counts for the 4090. Artifacts expire after
# a few days: run the workflow again to get a new one.
# Needs: ssh to the desk over the tailnet, gh, bao-mint-github, the Mac stunnel client
# (launchd com.chibifire.ninfer-tunnel) and BONSAI_DESK_API_KEY in ~/.hermes/.env.

const repo = "V-Sekai-fire/ninfer-all"
const desk = "ernest.lee@100.93.160.103"
const desk_ip = "100.93.160.103"
const desk_cn = "windows-229dd3.agents.weftspun"
const cards = {
    "4090": {arch: "89", tunnel: 8443, local: 18091}
    "3090": {arch: "86", tunnel: 8444, local: 18092}
}
const tunnel_conf = "~/.local/share/ninfer-tunnel/stunnel.conf"
const mac_cert_dir = "~/.bao-creds/mac-mini-fire-01b8f5"

def gh-token [] {
    if ($env.GH_TOKEN? | is-not-empty) { return $env.GH_TOKEN }
    let r = (^bao-mint-github V-Sekai-fire | complete)
    let hits = ([$r.stdout $r.stderr] | str join "\n" | lines | where $it =~ '^gh[sp]_')
    if ($hits | is-empty) { error make {msg: $"bao-mint-github returned no token: ($r.stderr | str trim)"} }
    $hits | first | split row " " | first
}

def env-value [name: string] {
    let row = (open --raw ("~/.hermes/.env" | path expand) | lines | where $it starts-with $"($name)=")
    if ($row | is-empty) { error make {msg: $"($name) is not in ~/.hermes/.env"} }
    $row | first | str substring (($name | str length) + 1)..
}

# Newest unexpired artifact for the card, as {id, name, run}.
def find-artifact [card: string, run?: int] {
    let arch = ($cards | get $card | get arch)
    let names = (if $arch == "89" { [dist-windows-sm89 dist-windows] } else { [$"dist-windows-sm($arch)"] })
    let path = if $run != null { $"repos/($repo)/actions/runs/($run)/artifacts" } else { $"repos/($repo)/actions/artifacts?per_page=100" }
    let all = (^gh api $path | from json | get artifacts)
    let hits = ($all | where {|a| $a.name in $names and not $a.expired } | sort-by created_at -r)
    if ($hits | is-empty) { error make {msg: $"no unexpired ($names | str join ' or ') artifact; run the V-Sekai Windows build workflow with arch ($arch)"} }
    let a = ($hits | first)
    {id: $a.id, name: $a.name, run: $a.workflow_run.id}
}

# The API answers an artifact download with a 302 to a short-lived blob URL that needs no token.
def artifact-url [id: int] {
    let token = (gh-token)
    let headers = (^curl -sI -H $"Authorization: Bearer ($token)" $"https://api.github.com/repos/($repo)/actions/artifacts/($id)/zip")
    let loc = ($headers | lines | where ($it | str lowercase | str starts-with "location:") | first | str substring 9.. | str trim)
    if ($loc | is-empty) { error make {msg: $"no download URL for artifact ($id)"} }
    $loc
}

def ensure-mac-tunnel [card: string] {
    let c = ($cards | get $card)
    let f = ($tunnel_conf | path expand)
    let text = (open --raw $f)
    let name = $"ninfer-desk-($card)"
    if ($text | str contains $"accept = 127.0.0.1:($c.local)") { return }
    let key = ($text | lines | where $it starts-with "key" | first)
    let section = ([
        ""
        $"[($name)]"
        "client = yes"
        $"accept = 127.0.0.1:($c.local)"
        $"connect = ($desk_ip):($c.tunnel)"
        $"sni = ($desk_cn)"
        $"cert = ($mac_cert_dir | path expand | path join cert-fullchain.pem)"
        $key
        $"CAfile = ('~/.bao.d/server-ca.pem' | path expand)"
        "verifyChain = yes"
        $"checkHost = ($desk_cn)"
        "sslVersionMin = TLSv1.3"
    ] | str join "\n") + "\n"
    $text + $section | save --force --raw $f
    ^launchctl kickstart -k $"gui/(^id -u | str trim)/com.chibifire.ninfer-tunnel"
    sleep 3sec
    print $"mac tunnel: added [($name)] on 127.0.0.1:($c.local)"
}

def http-code [url: string, key?: string] {
    if $key == null { ^curl -s -o /dev/null -w "%{http_code}" $url } else { ^curl -s -o /dev/null -w "%{http_code}" -H $"Authorization: Bearer ($key)" $url }
}

# Witnesses through the Mac end of the tunnel. Never print a /v1/models body: it echoes --api-key.
def check [card: string] {
    let base = $"http://127.0.0.1:($cards | get $card | get local)"
    let key = (env-value BONSAI_DESK_API_KEY)
    let no_key = (http-code $"($base)/v1/models")
    let bad_key = (http-code $"($base)/v1/models" "not-the-key")
    let good = (http-code $"($base)/v1/models" $key)
    let body = {model: "bonsai-27b", temperature: 0, max_tokens: 512, messages: [{role: user, content: "Write a Python function that parses an ISO-8601 date and returns the weekday name. Explain it."}]}
    let r = (^curl -s -H $"Authorization: Bearer ($key)" -H "Content-Type: application/json" $"($base)/v1/chat/completions" -d ($body | to json) | from json)
    let text = ($r.choices.0.message.content)
    let vision = (with-env {BONSAI_API_KEY: $key} { ^python3 ($env.FILE_PWD | path join vision_zero_shot_probe.py) $base | complete })
    let result = {
        card: $card
        no_key: $no_key, wrong_key: $bad_key, with_key: $good
        decode_tok_s: ($r.timings?.predicted_per_second? | default 0 | math round --precision 1)
        draft_accepted: $"($r.timings?.draft_n_accepted?)/($r.timings?.draft_n?)"
        answer_sha256: ($text | hash sha256 | str substring 0..15)
        answer_head: ($text | str substring 0..120)
        vision: ($vision.stdout | lines | where $it !~ '^(model|usage):' | str join " ")
    }
    print ($result | table -e)
    if $no_key != "401" or $bad_key != "401" or $good != "200" { error make {msg: "auth witness failed"} }
    if not ($result.vision =~ '(?i)red' and $result.vision =~ '(?i)blue' and $result.vision =~ '42') { error make {msg: "vision witness failed"} }
    $result
}

def main [card: string, --run: int, --keep-build] {
    if $card not-in ($cards | columns) { error make {msg: $"card must be one of ($cards | columns | str join ', ')"} }
    $env.GH_TOKEN = (gh-token)
    let here = $env.FILE_PWD
    ^ssh $desk "mkdir C:\\ninfer-scratch 2>nul & exit 0"
    ^scp -q ($here | path join install.nu) $"($desk):C:/ninfer-scratch/install.nu"
    if $keep_build {
        ^ssh $desk $"nu C:/ninfer-scratch/install.nu ($card)"
    } else {
        let a = (find-artifact $card $run)
        print $"($card): artifact ($a.name) from run ($a.run)"
        let tmp = (mktemp -t)
        artifact-url $a.id | save --force --raw $tmp
        ^scp -q $tmp $"($desk):C:/ninfer-scratch/u.txt"
        rm $tmp
        ^ssh $desk $"nu C:/ninfer-scratch/install.nu ($card) --url-file C:/ninfer-scratch/u.txt"
    }
    if $env.LAST_EXIT_CODE != 0 { error make {msg: "desk install failed"} }
    ensure-mac-tunnel $card
    check $card | ignore
}
