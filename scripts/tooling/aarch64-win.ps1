param([string]$Command, [string]$Selected, [switch]$Ci, [ValidateSet('dev', 'ci')][string]$Operation)
$ErrorActionPreference = 'Stop'
# -Operation names the operation as the Linux installers do; -Ci is its shorthand.
if ($Operation -eq 'ci') { $Ci = $true }
# Any host, including the macOS workstation, can name the inputs of an installation.
if ($Command -eq 'fingerprint' -and $args.Count -eq 0) {
    $fingerprintOperation = if ($Selected) { $Selected } elseif ($Ci) { 'ci' } else { 'dev' }
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:UV_PROJECT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $env:UV_LOCKED = '1'
    & uv run python "$PSScriptRoot/fingerprint.py" aarch64-win $fingerprintOperation
    exit $LASTEXITCODE
}
if ($Command -or $Selected -or $args.Count -ne 0) {
    throw 'Usage: scripts/tooling/aarch64-win.ps1 [-Ci|-Operation <dev|ci>] | fingerprint [<operation>]'
}
& "$PSScriptRoot/windows.ps1" -Platform aarch64-win -Ci:$Ci
