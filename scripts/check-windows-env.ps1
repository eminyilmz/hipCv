param(
    [switch]$RequireHip
)

$ErrorActionPreference = "SilentlyContinue"

function Find-CommandPath {
    param([string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return $null
    }

    return $command.Source
}

function Add-Check {
    param(
        [System.Collections.ArrayList]$Results,
        [string]$Name,
        [bool]$Found,
        [bool]$Required,
        [string]$Details
    )

    [void]$Results.Add([pscustomobject]@{
        Name = $Name
        Status = if ($Found) { "OK" } elseif ($Required) { "MISSING" } else { "OPTIONAL" }
        Required = if ($Required) { "yes" } else { "no" }
        Details = $Details
    })
}

function Find-VisualStudioInstall {
    $vswhereCandidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )

    foreach ($candidate in $vswhereCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            $installPath = & $candidate -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($installPath)) {
                return $installPath.Trim()
            }
        }
    }

    return $null
}

function Find-HipRoot {
    $roots = @()

    foreach ($name in @("HIP_PATH", "ROCM_PATH", "HIPSDK_ROOT", "ROCM_ROOT")) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            $roots += "$name=$value"
        }
    }

    $commonRoots = @(
        "C:\Program Files\AMD\ROCm",
        "C:\Program Files\AMD\HIP SDK"
    )

    foreach ($root in $commonRoots) {
        if (Test-Path -LiteralPath $root) {
            $children = Get-ChildItem -LiteralPath $root -Directory
            if ($children.Count -gt 0) {
                foreach ($child in $children) {
                    $roots += $child.FullName
                }
            } else {
                $roots += $root
            }
        }
    }

    return $roots
}

function Find-HipTool {
    param(
        [string]$Name,
        [array]$Roots
    )

    $fromPath = Find-CommandPath $Name
    if ($null -ne $fromPath) {
        return $fromPath
    }

    foreach ($root in $Roots) {
        $rootPath = $root
        if ($root -match "^[A-Z_]+=(.*)$") {
            $rootPath = $Matches[1]
        }

        $candidate = Join-Path (Join-Path $rootPath "bin") $Name
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return $null
}

function Find-DuplicatePathVariables {
    $pathLines = cmd /d /c set path 2>$null
    $pathNames = @()

    foreach ($line in $pathLines) {
        if ($line -match "^(.*?)=") {
            $name = $Matches[1]
            if ($name -ieq "path") {
                $pathNames += $name
            }
        }
    }

    return $pathNames
}

$results = New-Object System.Collections.ArrayList

$cmake = Find-CommandPath "cmake"
$cl = Find-CommandPath "cl"
$msbuild = Find-CommandPath "msbuild"
$gxx = Find-CommandPath "g++"
$clangxx = Find-CommandPath "clang++"
$vsInstall = Find-VisualStudioInstall
$hipRoots = Find-HipRoot
$hipInfo = Find-HipTool "hipInfo.exe" $hipRoots
$hipConfig = Find-HipTool "hipconfig.exe" $hipRoots
$pathNames = Find-DuplicatePathVariables

$hasCppToolchain = $null -ne $cl -or $null -ne $msbuild -or $null -ne $vsInstall -or $null -ne $gxx -or $null -ne $clangxx
$hasHipTools = $null -ne $hipInfo -and $null -ne $hipConfig
$hasDuplicatePathNames = $pathNames.Count -gt 1
$suggestedPreset = if ($vsInstall -match "\\18\\") { "windows-vs2026" } else { "windows-vs2022" }

Add-Check $results "CMake" ($null -ne $cmake) $true ($(if ($cmake) { $cmake } else { "cmake not found on PATH" }))
Add-Check $results "C++ toolchain" $hasCppToolchain $true ($(if ($hasCppToolchain) {
    $details = @()
    if ($cl) { $details += "cl=$cl" }
    if ($msbuild) { $details += "msbuild=$msbuild" }
    if ($vsInstall) { $details += "Visual Studio=$vsInstall" }
    if ($gxx) { $details += "g++=$gxx" }
    if ($clangxx) { $details += "clang++=$clangxx" }
    $details -join "; "
} else {
    "Visual Studio C++ Build Tools, g++, or clang++ not found"
}))
Add-Check $results "hipInfo" ($null -ne $hipInfo) ([bool]$RequireHip) ($(if ($hipInfo) { $hipInfo } else { "hipInfo not found on PATH" }))
Add-Check $results "hipconfig" ($null -ne $hipConfig) ([bool]$RequireHip) ($(if ($hipConfig) { $hipConfig } else { "hipconfig not found on PATH" }))
Add-Check $results "HIP SDK root" ($hipRoots.Count -gt 0) ([bool]$RequireHip) ($(if ($hipRoots.Count -gt 0) { $hipRoots -join "; " } else { "no common HIP SDK root found" }))
Add-Check $results "Duplicate PATH names" (-not $hasDuplicatePathNames) $false ($(if ($hasDuplicatePathNames) { "found: $($pathNames -join ', '); MSBuild may fail until the shell environment is cleaned" } else { "only one Path/PATH variable detected" }))

Write-Host ""
Write-Host "hipcv Windows environment check"
Write-Host "================================"
Write-Host ""
$results | Format-Table -AutoSize

$missingRequired = $results | Where-Object { $_.Required -eq "yes" -and $_.Status -eq "MISSING" }

Write-Host ""
if ($missingRequired.Count -gt 0) {
    Write-Host "Result: missing required tools." -ForegroundColor Red
    Write-Host ""
    Write-Host "For no-HIP development, install CMake and Visual Studio 2022 Build Tools with the C++ workload."
    Write-Host "For HIP development, install AMD HIP SDK for Windows and make sure hipInfo and hipconfig are available."
    exit 1
}

if ($hasDuplicatePathNames) {
    Write-Host "Note: duplicate PATH variable names can break MSBuild with: Key in dictionary: 'Path' Key being added: 'PATH'." -ForegroundColor Yellow
    Write-Host "A temporary child-shell workaround is to clear one casing before running CMake:"
    Write-Host "  cmd /c `"set Path=& set PATH=<your normal PATH>& cmake --preset windows-vs2026-no-hip`""
    Write-Host ""
}

if (-not $RequireHip -and -not $hasHipTools) {
    Write-Host "Result: no-HIP development environment is ready if the C++ toolchain above is usable." -ForegroundColor Yellow
    Write-Host "HIP tools were not found, so use the no-HIP preset first:"
    Write-Host "  cmake --preset $suggestedPreset-no-hip"
    Write-Host "  cmake --build --preset $suggestedPreset-no-hip-release"
    Write-Host "  ctest --preset $suggestedPreset-no-hip-release"
    exit 0
}

Write-Host "Result: required tools were found." -ForegroundColor Green
Write-Host ""
Write-Host "Suggested next commands:"
if ($RequireHip) {
    Write-Host "  cmake --preset $suggestedPreset"
    Write-Host "  cmake --build --preset $suggestedPreset-release"
    Write-Host "  ctest --preset $suggestedPreset-release"
} else {
    Write-Host "  cmake --preset $suggestedPreset-no-hip"
    Write-Host "  cmake --build --preset $suggestedPreset-no-hip-release"
    Write-Host "  ctest --preset $suggestedPreset-no-hip-release"
}
