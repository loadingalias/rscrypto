# Shared native Windows provisioning. Run in an elevated PowerShell session.
param([Parameter(Mandatory)][ValidateSet('aarch64-win', 'x86_64-win')][string]$Platform, [switch]$Ci)
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-StrictMode -Version Latest
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw 'This installer requires Windows.' }
$architecture = (Get-CimInstance Win32_Processor | Select-Object -First 1).Architecture
$expectedArchitecture = if ($Platform -eq 'aarch64-win') { 12 } else { 9 }
if ($architecture -ne $expectedArchitecture) { throw "$Platform requires native $expectedArchitecture hardware." }
$os = Get-CimInstance Win32_OperatingSystem
if ($Platform -eq 'aarch64-win') {
    if ($os.Caption -notmatch 'Windows 11 Enterprise') { throw 'Expected Windows 11 Enterprise ARM64.' }
} elseif ($os.Caption -notmatch 'Windows Server 2025') { throw 'Expected Windows Server 2025.' }
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run the installer from an elevated PowerShell session.'
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$catalogPath = Join-Path $repoRoot '.config/tooling.toml'
$prefix = Join-Path $env:LOCALAPPDATA 'rscrypto\tooling'
New-Item -ItemType Directory -Force $prefix | Out-Null
$temporary = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $temporary | Out-Null

function Invoke-Native {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed with exit code $LASTEXITCODE" }
}
function Get-PinnedDownload {
    param([string]$Url, [string]$Sha256, [string]$Destination)
    if ($Url -notmatch '^https://' -or $Sha256 -notmatch '^[0-9a-f]{64}$') { throw 'Invalid download pin.' }
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination
    if ((Get-FileHash -Algorithm SHA256 $Destination).Hash -ne $Sha256) {
        Remove-Item $Destination -Force
        throw "Checksum mismatch for $Url"
    }
}
function Install-Exe {
    param([string]$Path, [string[]]$Arguments)
    $process = Start-Process -FilePath $Path -ArgumentList $Arguments -Wait -PassThru
    if ($process.ExitCode -eq 3010) {
        throw 'Installation requires a reboot. Reboot and rerun this same installer.'
    }
    if ($process.ExitCode -ne 0) { throw "$Path failed with exit code $($process.ExitCode)" }
}

try {
    # Bootstrap only two scalar strings; Python's standard TOML parser reads the catalog afterward.
    $text = Get-Content -Raw $catalogPath
    $section = [regex]::Match($text, '(?ms)^\[' + [regex]::Escape($Platform) + '\.assets\.python\]\r?\n(?<body>.*?)(?=^\[|\z)')
    if (-not $section.Success) { throw 'Missing native Python bootstrap asset.' }
    $url = [regex]::Match($section.Groups['body'].Value, '(?m)^url = ("[^"\r\n]+")\r?$').Groups[1].Value | ConvertFrom-Json
    $sha = [regex]::Match($section.Groups['body'].Value, '(?m)^sha256 = ("[0-9a-f]{64}")\r?$').Groups[1].Value | ConvertFrom-Json
    $pythonDirectory = Join-Path $prefix ('python\' + $sha.Substring(0, 16))
    $python = Join-Path $pythonDirectory 'python.exe'
    if (-not (Test-Path $python)) {
        $archive = Join-Path $temporary 'python.zip'
        Get-PinnedDownload $url $sha $archive
        Expand-Archive -Path $archive -DestinationPath $pythonDirectory -Force
        Add-Content (Get-ChildItem $pythonDirectory -Filter 'python*._pth' | Select-Object -First 1).FullName $PSScriptRoot
    }
    $python3 = Join-Path $pythonDirectory 'python3.exe'
    if (-not (Test-Path $python3)) { New-Item -ItemType HardLink -Path $python3 -Target $python | Out-Null }
    $catalogHelper = Join-Path $PSScriptRoot 'catalog.py'
    $catalogJson = & $python $catalogHelper json
    if ($LASTEXITCODE -ne 0) { throw 'Unable to parse tooling catalog.' }
    $catalog = $catalogJson | ConvertFrom-Json
    Invoke-Native $python @($catalogHelper, 'validate')
    $native = $catalog.$Platform
    $toolchainHelper = Join-Path $PSScriptRoot '../lib/toolchain.py'
    $channel = & $python $toolchainHelper --target $native.'rust-host'
    if ($LASTEXITCODE -ne 0) { throw 'Unable to read rust-toolchain.toml.' }

    $channelFile = Join-Path $prefix ('vs-channel-' + $catalog.windows.'channel-sha256' + '.json')
    Get-PinnedDownload $catalog.windows.'channel-url' $catalog.windows.'channel-sha256' $channelFile
    $bootstrap = Join-Path $temporary 'vs_buildtools.exe'
    Get-PinnedDownload $catalog.windows.'bootstrap-url' $catalog.windows.'bootstrap-sha256' $bootstrap
    $vsPath = Join-Path $prefix ('vs-' + $catalog.windows.'visual-studio')
    $vsToolset = if ($Platform -eq 'aarch64-win') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' } else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
    $vsArguments = @('--quiet', '--wait', '--norestart', '--noUpdateInstaller',
        '--installPath', ('"' + $vsPath + '"'), '--installChannelUri', ('"' + $channelFile + '"'),
        '--channelUri', ('"' + $channelFile + '"'), '--add', $vsToolset,
        '--add', $catalog.windows.'sdk-component')
    if (-not $Ci -and $Platform -eq 'x86_64-win') { $vsArguments += @('--add', 'Microsoft.VisualStudio.Component.VC.ASAN') }
    if (Test-Path (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')) {
        $vsArguments = @('modify') + $vsArguments
    }
    Install-Exe $bootstrap $vsArguments
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $instances = & $vswhere -products Microsoft.VisualStudio.Product.BuildTools -format json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect Visual Studio installation.' }
    $instance = @($instances | Where-Object { $_.installationPath -eq $vsPath -and $_.installationVersion -eq $catalog.windows.'build-version' })
    if ($instance.Count -ne 1) { throw 'Visual Studio does not match the pinned build.' }
    Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    $vsArch = if ($Platform -eq 'aarch64-win') { 'arm64' } else { 'amd64' }
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=$vsArch -host_arch=$vsArch"
    $msvcBin = Split-Path (Get-Command cl.exe -CommandType Application).Source -Parent

    $gitInstaller = Join-Path $temporary 'git.exe'
    Get-PinnedDownload $native.assets.git.url $native.assets.git.sha256 $gitInstaller
    $gitDirectory = Join-Path $prefix ('git-' + $catalog.versions.git)
    Install-Exe $gitInstaller @('/VERYSILENT', '/NORESTART', '/NOCANCEL', '/SP-', ('/DIR="' + $gitDirectory + '"'))
    $binDirectory = Join-Path $prefix 'bin'
    New-Item -ItemType Directory -Force $binDirectory | Out-Null
    Get-PinnedDownload $native.assets.jq.url $native.assets.jq.sha256 (Join-Path $binDirectory 'jq.exe')
    # Keep Microsoft's link.exe ahead of Git's Unix link utility.
    $paths = @($msvcBin, $pythonDirectory, $binDirectory, (Join-Path $gitDirectory 'cmd'),
        (Join-Path $gitDirectory 'bin'), (Join-Path $gitDirectory 'usr\bin'))
    $archives = @('llvm', 'cmake', 'cargo-binstall')
    if (-not $Ci) { $archives += @('cargo-rail', 'powershell') }
    if ($Platform -eq 'x86_64-win') { $archives += 'nasm' }
    foreach ($name in $archives) {
        $directory = & $python $catalogHelper install-archive $Platform $name $prefix
        if ($LASTEXITCODE -ne 0) { throw "Unable to install $name" }
        $toolBin = if (Test-Path (Join-Path $directory 'bin')) { Join-Path $directory 'bin' } else { $directory }
        $paths += $toolBin
        if ($name -eq 'llvm') { $env:LIBCLANG_PATH = $toolBin }
    }
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
    $cargoBin = Join-Path $cargoHome 'bin'
    $paths += $cargoBin
    $env:PATH = (@($paths + ($env:PATH -split ';')) | Select-Object -Unique) -join ';'
    $linkerVariable = 'CARGO_TARGET_' + $native.'rust-host'.ToUpperInvariant().Replace('-', '_') + '_LINKER'
    [Environment]::SetEnvironmentVariable($linkerVariable, (Join-Path $msvcBin 'link.exe'), 'Process')
    $rustupInstaller = Join-Path $temporary 'rustup-init.exe'
    Get-PinnedDownload $native.assets.rustup.url $native.assets.rustup.sha256 $rustupInstaller
    Invoke-Native $rustupInstaller @('-y', '--no-modify-path', '--default-host', $native.'rust-host', '--default-toolchain', 'none')
    $rustArguments = @($toolchainHelper, '--install', $native.'rust-host')
    if (-not $Ci) {
        foreach ($component in $native.components) { $rustArguments += @('--component', $component) }
    }
    Invoke-Native $python $rustArguments
    Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
    Remove-Item Env:CARGO_ENCODED_RUSTFLAGS -ErrorAction SilentlyContinue
    $cargoTools = if ($Ci) { $catalog.ci.cargo } else { $native.cargo }
    foreach ($tool in $cargoTools) {
        Invoke-Native 'cargo' @("+$channel", 'binstall', '--locked', '--no-confirm', '--targets', $native.'rust-host', "$tool@$($catalog.cargo.$tool)")
    }
    $probeDirectory = Join-Path $temporary 'compiler-probe'
    New-Item -ItemType Directory -Force (Join-Path $probeDirectory 'src') | Out-Null
    Set-Content -Path (Join-Path $probeDirectory 'Cargo.toml') -Encoding ASCII -Value @(
        '[package]', 'name = "tooling_probe"', 'version = "0.0.0"', 'edition = "2024"')
    Set-Content -Path (Join-Path $probeDirectory 'src/main.rs') -Value 'fn main() {}' -Encoding ASCII
    Set-Content -Path (Join-Path $probeDirectory 'build.rs') -Value 'fn main() {}' -Encoding ASCII
    $probeCommands = @(
        'check:', "    cargo +$channel run --target $($native.'rust-host')")
    if ($Platform -eq 'x86_64-win') {
        Set-Content -Path (Join-Path $probeDirectory 'probe.asm') -Encoding ASCII -Value @(
            'section .text', 'global tooling_probe', 'tooling_probe:', '    ret')
        $probeCommands += '    nasm -f win64 probe.asm -o probe.obj'
    }
    Set-Content -Path (Join-Path $probeDirectory 'justfile') -Encoding ASCII -Value $probeCommands
    Invoke-Native 'just' @('--justfile', (Join-Path $probeDirectory 'justfile'), 'check')
    Invoke-Native 'clang' @('--version')
    Invoke-Native 'cmake' @('--version')
    if ($Platform -eq 'x86_64-win') { Invoke-Native 'nasm' @('-v') }
    if (-not $Ci) { Invoke-Native 'cargo' @("+$channel", 'rail', '--version') }
    Invoke-Native 'cargo' @("+$channel", 'nextest', '--version')

    # Persist the complete MSVC/SDK environment, not only the paths to installed executables.
    if (-not $Ci) {
        foreach ($name in @('PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'LIBCLANG_PATH', 'VSINSTALLDIR', 'VCINSTALLDIR', 'VCToolsInstallDir', 'WindowsSdkDir', 'WindowsSDKVersion', $linkerVariable)) {
            $value = [Environment]::GetEnvironmentVariable($name, 'Process')
            if ($value) { [Environment]::SetEnvironmentVariable($name, $value, 'User') }
        }
    }
    Write-Host "Installed $Platform tooling. The current process has the configured compiler environment."
} finally {
    Remove-Item -Recurse -Force $temporary -ErrorAction SilentlyContinue
}
