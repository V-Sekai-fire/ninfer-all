# Archive models/ and sources/ to the private Hugging Face dataset chibifire/ninfer-bonsai2-heretic,
# as a launchd oneshot. Install: nu v-sekai/hf-archive-job.nu install   Log: ~/Library/Logs/ninfer-hf-archive.log
# Check only:  nu v-sekai/hf-archive-job.nu reconcile   (exit 1 unless every local file is remote with the same sha256)
# `nu v-sekai/hf-archive-job.nu prune` reconciles, then deletes the local sources/ (models/ stays for serving).
# sources/heretic-v2-mlx-2bit is not archived: it is 3hs4n's public repo, needed only by parity_reference.py.
# Credits: one upload at a time. Backpressure: Nice 10, LowPriorityIO.

const label = "com.chibifire.ninfer-hf-archive"
const repo = "chibifire/ninfer-bonsai2-heretic"

const script_dir = path self .
def root [] { $script_dir | path dirname }
def plist-path [] { $env.HOME | path join Library LaunchAgents $"($label).plist" }
def token [] { ^security find-generic-password -s hermes.huggingface-token -w | str trim }

# Local data files as {path, size}: models/* and sources/**, minus hf download caches.
def local-files [] {
    cd (root)
    glob models/* --no-dir | append (glob sources/**/* --no-dir)
    | each {|f| $f | path relative-to (root) }
    | where {|p| not ($p | str contains "/.cache/") and not ($p | str starts-with "sources/heretic-v2-mlx-2bit") }
    | each {|p| {path: $p, size: (ls $p | get 0.size | into int)} }
}

# Remote files as {path, size, sha256}, following the tree listing's pages.
def remote-files [] {
    let auth = {Authorization: $"Bearer (token)"}
    mut url = $"https://huggingface.co/api/datasets/($repo)/tree/main?recursive=true"
    mut out = []
    loop {
        let r = (http get --full -H $auth $url)
        $out = ($out | append ($r.body | where type == file | each {|f|
            {path: $f.path, size: $f.size, sha256: ($f.lfs?.oid? | default "")} }))
        let link = ($r.headers.response | where name == link | get -o 0.value | default "")
        let next = ($link | parse -r '<(?<u>[^>]+)>;\s*rel="next"')
        if ($next | is-empty) { break }
        $url = ($next | get 0.u)
    }
    $out
}

def "main reconcile" [] {
    let remote = (remote-files)
    let rows = (local-files | each {|f|
        let r = ($remote | where path == $f.path | get -o 0)
        let sha = (^shasum -a 256 ((root) | path join $f.path) | split row " " | get 0)
        {path: $f.path, size: $f.size,
         ok: ($r != null and $r.size == $f.size and ($r.sha256 == $sha or ($r.sha256 == "" and $f.size < 10_000_000)))}
    })
    let private = (http get -H {Authorization: $"Bearer (token)"} $"https://huggingface.co/api/datasets/($repo)" | get private)
    let bad = ($rows | where not ok)
    print $"($rows | length) local files, ($bad | length) missing or different; private=($private)"
    if ($bad | is-not-empty) { print $bad }
    if ($bad | is-not-empty) or not $private { exit 1 }
}

def "main run" [] {
    print $"[job] start (date now | format date '%FT%T')"
    $env.HF_TOKEN = (token)
    $env.HF_HUB_DISABLE_PROGRESS_BARS = "1"
    cd (root)
    ^hf repos create $repo --repo-type dataset --private --exist-ok
    ^hf upload $repo v-sekai/hf-archive-card.md README.md --repo-type dataset --commit-message "Describe the archive"
    # Xet-backed and resumable: rerunning the job continues an interrupted upload.
    ^hf upload $repo models models --repo-type dataset --commit-message "Add the converted artifact"
    ^hf upload $repo sources sources --repo-type dataset --exclude "**/.cache/**" "heretic-v2-mlx-2bit/**" --commit-message "Add the conversion sources"
    main reconcile
    print $"[job] done (date now | format date '%FT%T')"
}

def "main prune" [] {
    main reconcile
    cd (root)
    let freed = (local-files | where {|f| $f.path | str starts-with "sources/" } | get size | math sum)
    rm -r sources
    print $"removed sources/ \(($freed | into filesize)\); models/ kept for serving"
    print $"restore: hf download ($repo) --repo-type dataset --include 'sources/*' --local-dir ."
}

def "main install" [] {
    let log = ($env.HOME | path join Library Logs ninfer-hf-archive.log)
    let pixi = (which pixi | get 0.path)
    let nu_bin = (which nu | get 0.path)
    {
        Label: $label
        ProgramArguments: [$pixi run --manifest-path ((root) | path join pixi.toml) --locked
            $nu_bin ((root) | path join v-sekai hf-archive-job.nu) run]
        WorkingDirectory: (root)
        RunAtLoad: true
        KeepAlive: false
        Nice: 10
        LowPriorityIO: true
        StandardOutPath: $log
        StandardErrorPath: $log
        EnvironmentVariables: {PATH: $"($pixi | path dirname):($nu_bin | path dirname):/usr/bin:/bin:/usr/sbin:/sbin", PYTHONUNBUFFERED: "1"}
    } | to json | save -f (plist-path)
    ^plutil -convert xml1 (plist-path)
    let uid = (^id -u | str trim)
    ^launchctl bootout $"gui/($uid)/($label)" | complete | ignore
    for _ in 1..20 {
        if (^launchctl print $"gui/($uid)/($label)" | complete | get exit_code) != 0 { break }
        sleep 500ms
    }
    ^launchctl bootstrap $"gui/($uid)" (plist-path)
    print $"started ($label); log ($log)"
}

def main [] { print "usage: nu v-sekai/hf-archive-job.nu install|run|reconcile|prune" }
