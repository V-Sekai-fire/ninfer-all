# MSVC x64 environment from the latest Visual Studio with the C++ tools.

# The environment vcvars64.bat produces, as a record. cmd reports PATH as "Path" or "PATH".
export def vcvars-env [] {
    let vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    let vs = (^$vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | str trim)
    let bat = ($vs | path join 'VC\Auxiliary\Build\vcvars64.bat')
    ^cmd /c $'"($bat)" >nul && set'
    | lines
    | parse '{name}={value}'
    | where name not-in [PWD FILE_PWD CURRENT_FILE]
    | reduce -f {} {|it, acc| $acc | upsert $it.name $it.value }
}

# Load MSVC into the caller's environment. The pixi PATH entries stay first, so the pixi
# CMake, Ninja and CUDA win over the copies Visual Studio ships. VCPKG_ROOT is dropped,
# because vcvars points it at the tree bundled with Visual Studio.
export def --env load-msvc [] {
    let vc = (vcvars-env)
    let path_key = ($vc | columns | where {|c| ($c | str downcase) == path } | first)
    let vc_path = ($vc | get $path_key | split row ';')
    let current = ($env.Path? | default $env.PATH? | default [])
    let current = if ($current | describe) == string { $current | split row ';' } else { $current }
    let rest = ($vc | reject $path_key | reject -i VCPKG_ROOT)
    load-env $rest
    $env.Path = ($current | append $vc_path | uniq)
}
