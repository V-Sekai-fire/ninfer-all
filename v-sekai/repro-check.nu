# Prove the build is reproducible: build the same commit twice in separate build trees and
# compare SHA256SUMS. Run from the fork root: pixi run --locked -e build nu v-sekai/repro-check.nu

def main [] {
    let root = ($env.FILE_PWD | path dirname)
    cd $root
    let os = $nu.os-info.name
    let sums = ($root | path join dist $os SHA256SUMS)
    mut results = []
    for name in [repro-a repro-b] {
        rm -rf ($root | path join $"($name)-($os)")
        ^nu v-sekai/build.nu --build-dir $name
        if $env.LAST_EXIT_CODE != 0 { error make {msg: $"Build ($name) failed"} }
        $results = ($results | append (open --raw $sums))
    }
    let a = ($results | first | lines)
    let b = ($results | last | lines)
    let differ = ($a | zip $b | where {|p| $p.0 != $p.1 })
    if ($differ | is-not-empty) or (($a | length) != ($b | length)) {
        $differ | each {|p| print $"a: ($p.0)\nb: ($p.1)" }
        error make {msg: "The two builds differ"}
    }
    print $"Reproducible: ($a | length) files identical"
}
