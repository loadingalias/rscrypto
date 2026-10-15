# Shared native Windows provisioning. Run in an elevated PowerShell session.
param([Parameter(Mandatory)][ValidateSet('aarch64-win', 'x86_64-win')][string]$Platform, [switch]$Ci, [switch]$CiBench, [switch]$CiCt)
$ErrorActionPreference = 'Stop'
if (([int]$Ci.IsPresent + [int]$CiBench.IsPresent + [int]$CiCt.IsPresent) -gt 1) { throw 'Select one CI profile.' }
$operation = if ($CiCt) { 'ci-ct' } elseif ($CiBench) { 'ci-bench' } elseif ($Ci) { 'ci' } else { 'dev' }
$Ci = $Ci -or $CiBench -or $CiCt
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
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        Write-Host "Downloading $Url (attempt $attempt/4)"
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination -TimeoutSec 300 -ErrorAction Stop
            break
        } catch {
            Remove-Item $Destination -Force -ErrorAction SilentlyContinue
            if ($attempt -eq 4) { throw }
            Write-Warning "Download failed for ${Url}: $($_.Exception.Message)"
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
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
function Get-PinnedVisualStudio {
    param([string]$Path, [string]$Version, [string[]]$Components)
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return }
    $instances = & $vswhere -products Microsoft.VisualStudio.Product.BuildTools -requires $Components -format json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect Visual Studio installation.' }
    $instances | Where-Object { $_.installationPath -eq $Path -and $_.installationVersion -eq $Version }
}
function Test-Pinned {
    param([string]$Python, [string]$Helper, [string[]]$Names)
    # Mismatches are reported on stderr, which Windows PowerShell would raise as errors.
    $ErrorActionPreference = 'Continue'
    & $Python $Helper require @Names 2>&1 | Out-Null
    $LASTEXITCODE -eq 0
}

try {
    # Bootstrap only the two scalar strings of the uv archive; uv then runs the pinned Python.
    $text = Get-Content -Raw $catalogPath
    $section = [regex]::Match($text, '(?ms)^\[' + [regex]::Escape($Platform) + '\.assets\.uv\]\r?\n(?<body>.*?)(?=^\[|\z)')
    if (-not $section.Success) { throw 'Missing native uv bootstrap asset.' }
    $url = [regex]::Match($section.Groups['body'].Value, '(?m)^url = ("[^"\r\n]+")\r?$').Groups[1].Value | ConvertFrom-Json
    $sha = [regex]::Match($section.Groups['body'].Value, '(?m)^sha256 = ("[0-9a-f]{64}")\r?$').Groups[1].Value | ConvertFrom-Json
    # The layout and receipt match catalog.py install-archive.
    $uvDirectory = Join-Path $prefix ('uv\' + $sha.Substring(0, 16))
    $uv = Join-Path $uvDirectory 'uv.exe'
    if (-not (Test-Path (Join-Path $uvDirectory '.rscrypto-installed'))) {
        $archive = Join-Path $temporary 'uv.zip'
        Get-PinnedDownload $url $sha $archive
        Remove-Item -Recurse -Force $uvDirectory -ErrorAction SilentlyContinue
        Expand-Archive -Path $archive -DestinationPath $uvDirectory -Force -ErrorAction Stop
        Set-Content -Path (Join-Path $uvDirectory '.rscrypto-installed') -Value $sha -Encoding ascii
    }
    $env:UV_PROJECT = $repoRoot
    $env:UV_LOCKED = '1'
    Push-Location $repoRoot
    # `--default` also names it python3 in uv's bin directory: dev-machines runs python3 on the host.
    # Precompiled, the standard library imports without recompiling on every helper start.
    try { Invoke-Native $uv @('python', 'install', '--default', '--compile-bytecode') } finally { Pop-Location }
    $pythonBin = & $uv python dir --bin
    if ($LASTEXITCODE -ne 0) { throw 'Unable to locate the uv Python executables.' }
    # uv selects the pinned interpreter once; each helper then starts without uv's project check.
    $python = & $uv run python -c 'import sys; print(sys.executable)'
    if ($LASTEXITCODE -ne 0) { throw 'Unable to select the pinned Python.' }
    $catalogHelper = Join-Path $PSScriptRoot 'catalog.py'
    $catalogJson = & $python $catalogHelper json
    if ($LASTEXITCODE -ne 0) { throw 'Unable to parse tooling catalog.' }
    $catalog = $catalogJson | ConvertFrom-Json
    Invoke-Native $python @($catalogHelper, 'validate')
    $native = $catalog.$Platform
    $toolchainHelper = Join-Path $PSScriptRoot '../lib/toolchain.py'
    $channel = & $python $toolchainHelper
    if ($LASTEXITCODE -ne 0) { throw 'Unable to read rust-toolchain.toml.' }
    $fingerprint = & $python (Join-Path $PSScriptRoot 'fingerprint.py') $Platform $operation
    if ($LASTEXITCODE -ne 0) { throw 'Unable to fingerprint the installation inputs.' }
    $fingerprintFile = Join-Path $prefix "fingerprint.$operation"
    # A complete run records the fingerprint of its inputs. While it matches, a converged
    # host or image republishes its environment for this operation and probes the pins.
    $current = (Test-Path $fingerprintFile) -and (Get-Content -Raw $fingerprintFile).Trim() -eq $fingerprint

    $vsPath = Join-Path $prefix ('vs-' + $catalog.windows.'visual-studio')
    $vsToolset = if ($Platform -eq 'aarch64-win') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' } else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
    $vsComponents = @($vsToolset, $catalog.windows.'sdk-component')
    if (-not $Ci -and $Platform -eq 'x86_64-win') { $vsComponents += 'Microsoft.VisualStudio.Component.VC.ASAN' }
    $vsVersion = $catalog.windows.'build-version'
    # vswhere reports the pinned build only while it holds every selected component.
    if (@(Get-PinnedVisualStudio -Path $vsPath -Version $vsVersion -Components $vsComponents).Count -ne 1) {
        $channelFile = Join-Path $prefix ('vs-channel-' + $catalog.windows.'channel-sha256' + '.json')
        Get-PinnedDownload $catalog.windows.'channel-url' $catalog.windows.'channel-sha256' $channelFile
        $bootstrap = Join-Path $temporary 'vs_buildtools.exe'
        Get-PinnedDownload $catalog.windows.'bootstrap-url' $catalog.windows.'bootstrap-sha256' $bootstrap
        $vsArguments = @('--quiet', '--wait', '--norestart', '--noUpdateInstaller',
            '--installPath', ('"' + $vsPath + '"'), '--installChannelUri', ('"' + $channelFile + '"'),
            '--channelUri', ('"' + $channelFile + '"'))
        foreach ($component in $vsComponents) { $vsArguments += @('--add', $component) }
        if (Test-Path (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')) {
            $vsArguments = @('modify') + $vsArguments
        }
        Install-Exe $bootstrap $vsArguments
        if (@(Get-PinnedVisualStudio -Path $vsPath -Version $vsVersion -Components $vsComponents).Count -ne 1) {
            throw 'Visual Studio does not match the pinned build.'
        }
    }
    # Module commands ignore the caller's preference in Windows PowerShell.
    Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll') -ErrorAction Stop
    $vsArch = if ($Platform -eq 'aarch64-win') { 'arm64' } else { 'amd64' }
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=$vsArch -host_arch=$vsArch" -ErrorAction Stop
    # A repeated run already has the toolset on PATH from the persisted user environment.
    $msvcBin = Split-Path (Get-Command cl.exe -CommandType Application | Select-Object -First 1).Source -Parent

    # Git and jq report their own versions; only a missing or different pin is installed.
    $gitDirectory = Join-Path $prefix ('git-' + $catalog.versions.git)
    $git = Join-Path $gitDirectory 'cmd\git.exe'
    if (-not ((Test-Path $git) -and (& $git --version) -eq "git version $($catalog.versions.git)")) {
        $gitInstaller = Join-Path $temporary 'git.exe'
        Get-PinnedDownload $native.assets.git.url $native.assets.git.sha256 $gitInstaller
        Install-Exe $gitInstaller @('/VERYSILENT', '/NORESTART', '/NOCANCEL', '/SP-', ('/DIR="' + $gitDirectory + '"'))
    }
    $bash = Join-Path $gitDirectory 'bin\bash.exe'
    if (-not (Test-Path $bash)) { throw 'Git for Windows did not install bash.exe.' }
    $env:RSCRYPTO_BASH = $bash
    $binDirectory = Join-Path $prefix 'bin'
    New-Item -ItemType Directory -Force $binDirectory | Out-Null
    $jq = Join-Path $binDirectory 'jq.exe'
    if (-not $CiCt -and -not ((Test-Path $jq) -and (& $jq --version) -eq "jq-$($catalog.versions.jq)")) {
        Get-PinnedDownload $native.assets.jq.url $native.assets.jq.sha256 $jq
    }
    # Keep Microsoft's link.exe ahead of Git's Unix link utility.
    $paths = @($msvcBin, $uvDirectory, $pythonBin, $binDirectory, (Join-Path $gitDirectory 'cmd'),
        (Join-Path $gitDirectory 'bin'), (Join-Path $gitDirectory 'usr\bin'))
    $archives = if ($CiCt) { @('cargo-binstall') } else { @('llvm', 'cmake', 'cargo-binstall') }
    if (-not $Ci) {
        $archives += 'powershell'
        if ($native.assets.PSObject.Properties.Name -contains 'cargo-rail') { $archives += 'cargo-rail' }
    }
    if ($Platform -eq 'x86_64-win' -and -not $CiCt) { $archives += 'nasm' }
    foreach ($name in $archives) {
        $directory = & $python $catalogHelper install-archive $Platform $name $prefix
        if ($LASTEXITCODE -ne 0) { throw "Unable to install $name" }
        $toolBin = if (Test-Path (Join-Path $directory 'bin')) { Join-Path $directory 'bin' } else { $directory }
        $paths += $toolBin
        if ($name -eq 'llvm') { $env:LIBCLANG_PATH = $toolBin }
    }
    if ($Platform -eq 'x86_64-win' -and -not $CiCt) {
        # Vendored OpenSSL needs a native Perl; Git's MSYS Perl cannot configure it.
        # Select it only for OpenSSL builds and keep Git's tools first on PATH.
        $perlDirectory = & $python $catalogHelper install-archive $Platform 'strawberry-perl' $prefix
        if ($LASTEXITCODE -ne 0) { throw 'Unable to install strawberry-perl' }
        $env:OPENSSL_SRC_PERL = Join-Path $perlDirectory 'perl\bin\perl.exe'
        if (-not (Test-Path $env:OPENSSL_SRC_PERL)) { throw 'Strawberry Perl did not install perl.exe.' }
    }
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
    $cargoBin = Join-Path $cargoHome 'bin'
    $paths += $cargoBin
    $env:PATH = (@($paths + ($env:PATH -split ';')) | Select-Object -Unique) -join ';'
    $linkerVariable = 'CARGO_TARGET_' + $native.'rust-host'.ToUpperInvariant().Replace('-', '_') + '_LINKER'
    [Environment]::SetEnvironmentVariable($linkerVariable, (Join-Path $msvcBin 'link.exe'), 'Process')
    $env:RUSTUP_TOOLCHAIN = $channel
    Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
    Remove-Item Env:CARGO_ENCODED_RUSTFLAGS -ErrorAction SilentlyContinue
    $cargoTools = if ($CiCt) { $catalog.'ci-ct'.cargo } elseif ($CiBench) { $catalog.'ci-bench'.cargo } elseif ($Ci) { $catalog.ci.cargo } else { $native.cargo }
    if ($current -and -not (Test-Pinned -Python $python -Helper $catalogHelper -Names (@('rustup') + $cargoTools))) {
        Write-Warning "Tooling fingerprint $fingerprint matches, but a pinned tool is missing; converging."
        $current = $false
    }
    if (-not (Test-Pinned -Python $python -Helper $catalogHelper -Names @('rustup'))) {
        $rustupInstaller = Join-Path $temporary 'rustup-init.exe'
        Get-PinnedDownload $native.assets.rustup.url $native.assets.rustup.sha256 $rustupInstaller
        Invoke-Native $rustupInstaller @('-y', '--no-modify-path', '--default-host', $native.'rust-host', '--default-toolchain', 'none')
    }
    if (-not $current) {
        $rustArguments = @($toolchainHelper, '--install')
        if (-not $Ci) {
            foreach ($component in $native.components) { $rustArguments += @('--component', $component) }
        }
        if ($CiBench -or $CiCt) {
            $components = if ($CiCt) { $catalog.'ci-ct'.components } else { $catalog.'ci-bench'.components }
            $minimalArguments = @('toolchain', 'install', $channel, '--profile', 'minimal')
            foreach ($component in $components) { $minimalArguments += @('--component', $component) }
            Invoke-Native 'rustup' $minimalArguments
        } else {
            Invoke-Native $python $rustArguments
        }
        foreach ($tool in $cargoTools) {
            if (Test-Pinned -Python $python -Helper $catalogHelper -Names @($tool)) { continue }
            Invoke-Native 'cargo' @("+$channel", 'binstall', '--locked', '--no-confirm', '--targets', $native.'rust-host', "$tool@$($catalog.cargo.$tool)")
        }
        if (-not $Ci -and $native.assets.PSObject.Properties.Name -notcontains 'cargo-rail') {
            Invoke-Native 'cargo' @("+$channel", 'install', '--locked', '--version', $catalog.versions.'cargo-rail', 'cargo-rail')
        }
        foreach ($tool in $cargoTools) {
            if ($tool.StartsWith('cargo-')) {
                Invoke-Native 'cargo' @("+$channel", $tool.Substring(6), '--version')
            } elseif ($tool -eq 'ripgrep') {
                Invoke-Native 'rg' @('--version')
            } else {
                Invoke-Native $tool @('--version')
            }
        }
    }
    # A converged installation compiled this probe when it recorded its fingerprint.
    if (-not $current) {
        $probeDirectory = Join-Path $temporary 'compiler-probe'
        New-Item -ItemType Directory -Force (Join-Path $probeDirectory 'src') | Out-Null
        Set-Content -Path (Join-Path $probeDirectory 'Cargo.toml') -Encoding ASCII -Value @(
            '[package]', 'name = "tooling_probe"', 'version = "0.0.0"', 'edition = "2024"')
        Set-Content -Path (Join-Path $probeDirectory 'src/main.rs') -Value 'fn main() {}' -Encoding ASCII
        Set-Content -Path (Join-Path $probeDirectory 'build.rs') -Value 'fn main() {}' -Encoding ASCII
        $probeCommands = @(
            'check:', "    cargo +$channel run --target $($native.'rust-host')")
        if ($Platform -eq 'x86_64-win' -and -not $CiCt) {
            Set-Content -Path (Join-Path $probeDirectory 'probe.asm') -Encoding ASCII -Value @(
                'section .text', 'global tooling_probe', 'tooling_probe:', '    ret')
            $probeCommands += '    nasm -f win64 probe.asm -o probe.obj'
        }
        Set-Content -Path (Join-Path $probeDirectory 'justfile') -Encoding ASCII -Value $probeCommands
        Invoke-Native 'just' @('--justfile', (Join-Path $probeDirectory 'justfile'), 'check')
    }
    if (-not $CiCt) {
        Invoke-Native $bash @('--version')
        Invoke-Native 'clang' @('--version')
        Invoke-Native 'cmake' @('--version')
        if ($Platform -eq 'x86_64-win') {
            Invoke-Native 'nasm' @('-v')
            # OpenSSL's Configure loads this module; Git's Perl lacks it.
            Invoke-Native $env:OPENSSL_SRC_PERL @('-MLocale::Maketext::Simple', '-e', '1')
        }
    }
    if (-not $Ci) { Invoke-Native 'cargo' @("+$channel", 'rail', '--version') }

    # Persist the complete MSVC/SDK environment, not only the paths to installed executables.
    if (-not $Ci) {
        foreach ($name in @('PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'LIBCLANG_PATH', 'OPENSSL_SRC_PERL', 'RSCRYPTO_BASH', 'VSINSTALLDIR', 'VCINSTALLDIR', 'VCToolsInstallDir', 'WindowsSdkDir', 'WindowsSDKVersion', $linkerVariable)) {
            $value = [Environment]::GetEnvironmentVariable($name, 'Process')
            if ($value) { [Environment]::SetEnvironmentVariable($name, $value, 'User') }
        }
    }
    if ($current) {
        Write-Output "Tooling fingerprint $fingerprint matches; the current process has the configured compiler environment."
    } else {
        Set-Content -Path $fingerprintFile -Value $fingerprint -Encoding ascii
        Write-Host "Installed $Platform tooling. The current process has the configured compiler environment."
    }
} finally {
    Remove-Item -Recurse -Force $temporary -ErrorAction SilentlyContinue
}
