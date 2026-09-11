param([switch]$Ci, [switch]$CiBench, [switch]$CiCt)
$ErrorActionPreference = 'Stop'
if ($args.Count -ne 0) { throw 'Usage: scripts/tooling/x86_64-win.ps1 [-Ci|-CiBench|-CiCt]' }
& "$PSScriptRoot/windows.ps1" -Platform x86_64-win -Ci:$Ci -CiBench:$CiBench -CiCt:$CiCt
