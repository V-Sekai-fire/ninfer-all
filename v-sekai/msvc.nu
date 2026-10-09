# MSVC x64 environment from the latest Visual Studio with the C++ tools.

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
