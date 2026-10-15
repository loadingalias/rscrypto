param([string]$Command, [string]$Selected, [switch]$Ci, [switch]$CiBench, [switch]$CiCt,
    [ValidateSet('dev', 'ci', 'ci-bench', 'ci-ct')][string]$Operation)
$ErrorActionPreference = 'Stop'
# -Operation names the operation as the Linux installers do; the switches are its shorthand.
switch ($Operation) { 'ci' { $Ci = $true } 'ci-bench' { $CiBench = $true } 'ci-ct' { $CiCt = $true } }
# Any host, including the macOS workstation, can name the inputs of an installation.
if ($Command -eq 'fingerprint' -and $args.Count -eq 0) {
    $fingerprintOperation = if ($Selected) { $Selected } elseif ($CiCt) { 'ci-ct' } elseif ($CiBench) { 'ci-bench' } elseif ($Ci) { 'ci' } else { 'dev' }
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:UV_PROJECT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $env:UV_LOCKED = '1'
    & uv run python "$PSScriptRoot/fingerprint.py" x86_64-win $fingerprintOperation
    exit $LASTEXITCODE
}
if ($Command -or $Selected -or $args.Count -ne 0) {
    throw 'Usage: scripts/tooling/x86_64-win.ps1 [-Ci|-CiBench|-CiCt|-Operation <dev|ci|ci-bench|ci-ct>] | fingerprint [<operation>]'
}
& "$PSScriptRoot/windows.ps1" -Platform x86_64-win -Ci:$Ci -CiBench:$CiBench -CiCt:$CiCt
