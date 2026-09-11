$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Load the production function without provisioning the host.
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'windows.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$function = $ast.Find({ param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-PinnedDownload'
}, $true)
. ([scriptblock]::Create($function.Extent.Text))

function Invoke-WebRequest {
    param([switch]$UseBasicParsing, $Uri, $OutFile, $TimeoutSec, $ErrorAction)
    $script:attempts++
    if (Test-Path $OutFile) { throw 'Partial download survived a failed attempt.' }
    if ($script:attempts -le $script:failures) {
        [IO.File]::WriteAllText($OutFile, 'partial')
        throw [IO.IOException]::new('connection reset')
    }
    [IO.File]::WriteAllText($OutFile, 'abc')
}
function Start-Sleep {
    param([int]$Seconds)
    $script:delays += $Seconds
}

$destination = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
$sha = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
try {
    foreach ($case in @('recovery', 'exhaustion', 'checksum')) {
        $script:attempts = 0
        $script:delays = @()
        $script:failures = switch ($case) { recovery { 3 } exhaustion { 4 } checksum { 0 } }
        $expectedHash = if ($case -eq 'checksum') { '0' * 64 } else { $sha }
        $failure = $null
        try { Get-PinnedDownload 'https://example.invalid/asset' $expectedHash $destination }
        catch { $failure = $_.Exception.Message }
        switch ($case) {
            recovery {
                if ($failure -or $script:attempts -ne 4 -or ($script:delays -join ',') -ne '2,4,6') {
                    throw "Recovery failed: $failure"
                }
                if ([IO.File]::ReadAllText($destination) -cne 'abc') { throw 'Incorrect downloaded bytes.' }
            }
            exhaustion {
                if ($failure -ne 'connection reset' -or $script:attempts -ne 4 -or
                    ($script:delays -join ',') -ne '2,4,6' -or (Test-Path $destination)) {
                    throw "Retry exhaustion failed: $failure"
                }
            }
            checksum {
                if ($failure -notlike 'Checksum mismatch*' -or $script:attempts -ne 1 -or
                    $script:delays.Count -ne 0 -or (Test-Path $destination)) {
                    throw "Checksum rejection failed: $failure"
                }
            }
        }
        Remove-Item $destination -Force -ErrorAction SilentlyContinue
    }
} finally {
    Remove-Item $destination -Force -ErrorAction SilentlyContinue
}
