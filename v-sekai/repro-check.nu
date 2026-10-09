# Cheap reproducibility check on an existing build: delete a sample of objects and the
# programs, rebuild just those, and require byte-identical output. A nondeterministic
# compiler or linker step (timestamps, random IDs, embedded paths) shows up as a changed hash.
# Run after build.nu: pixi run --locked -e build nu v-sekai/repro-check.nu --build-dir build

def main [--build-dir: string = "build", --sample: int = 6] {
    let root = ($env.FILE_PWD | path dirname)
    cd $root
    let os = $nu.os-info.name
    let tree = ($root | path join $"($build_dir)-($os)")
    let sums = ($root | path join dist $os SHA256SUMS)
    let before = (open --raw $sums)

    # Every Nth object, CUDA and C++ alike, so the sample is spread and repeatable.
    let objects = (glob ($tree | path join "**" "*.{obj,o}") | sort)
    let step = ([1 (($objects | length) // $sample)] | math max)
    let picked = ($objects | enumerate | where {|o| $o.index mod $step == 0 } | first $sample | get item)
    let suffix = if $os == windows { ".exe" } else { "" }
    let programs = ([ninfer-serve ninfer-calibrate] | each {|p| $tree | path join apps $"($p)($suffix)" })
    let old = ($picked | each {|f| {file: $f, sha: (open --raw $f | hash sha256)} })
    rm -f ...$picked ...$programs

    ^nu v-sekai/build.nu --build-dir $build_dir
    if $env.LAST_EXIT_CODE != 0 { error make {msg: "Rebuild failed"} }

    let objects_changed = ($old | where {|o| (open --raw $o.file | hash sha256) != $o.sha } | get file)
    let after = (open --raw $sums)
    let files_changed = ($before | lines | zip ($after | lines) | where {|p| $p.0 != $p.1 } | each {|p| $p.1 | split row "  " | last })
    if ($objects_changed | is-not-empty) or ($files_changed | is-not-empty) {
        print $"Changed objects: ($objects_changed)"
        print $"Changed files: ($files_changed)"
        error make {msg: "The rebuild is not byte-identical"}
    }
    print $"Reproducible: ($picked | length) recompiled objects and (($after | lines | length)) staged files identical"
}
