# SwiftImpute Build Script for Windows
# Requires: Visual Studio 2022, CUDA Toolkit 12.0+, CMake 3.20+

param(
    [string]$Config = "Release",
    [string]$Arch = "89",
    [int]$Jobs = 8,
    [switch]$Clean,
    [switch]$Test,
    [switch]$Benchmark,
    [switch]$Verbose,
    [switch]$FastMath
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { param($msg) Write-Host $msg -ForegroundColor Green }
function Write-Info { param($msg) Write-Host $msg -ForegroundColor Cyan }
function Write-Warning { param($msg) Write-Host $msg -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host $msg -ForegroundColor Red }

Write-Info "SwiftImpute Build Script"
Write-Info "========================="
Write-Host ""

# Validate prerequisites
Write-Info "Checking prerequisites..."

# Check CMake
try {
    $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
    Write-Success "Found: $cmakeVersion"
} catch {
    Write-Error "CMake not found. Please install CMake 3.20 or later."
    exit 1
}

# Check CUDA
try {
    $cudaVersion = nvcc --version 2>&1 | Select-String "release" | Select-Object -First 1
    Write-Success "Found: $cudaVersion"
} catch {
    Write-Error "CUDA not found. Please install CUDA Toolkit 12.0 or later."
    exit 1
}

# Check for Visual Studio
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    if ($vsPath) {
        Write-Success "Found Visual Studio at: $vsPath"
    }
} else {
    Write-Warning "Visual Studio detection failed. Build may fail."
}

Write-Host ""

# Configuration
$buildDir = "build"
$sourceDir = Get-Location

Write-Info "Build Configuration:"
Write-Host "  Config: $Config"
Write-Host "  Architecture: $Arch"
Write-Host "  Parallel jobs: $Jobs"
Write-Host "  Fast math: $FastMath"
Write-Host ""

# Clean build directory
if ($Clean) {
    Write-Info "Cleaning build directory..."
    if (Test-Path $buildDir) {
        Remove-Item -Recurse -Force $buildDir
        Write-Success "Build directory cleaned"
    }
}

# Create build directory
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# CMake configuration
Write-Info "Configuring with CMake..."

$cmakeArgs = @(
    "-B", $buildDir,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCUDA_ARCHITECTURES=$Arch"
)

if ($FastMath) {
    $cmakeArgs += "-DUSE_FAST_MATH=ON"
}

if ($Verbose) {
    $cmakeArgs += "-DCMAKE_VERBOSE_MAKEFILE=ON"
}

try {
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    Write-Success "Configuration successful"
} catch {
    Write-Error "Configuration failed: $_"
    exit 1
}

Write-Host ""

# Build
Write-Info "Building project..."

$buildArgs = @(
    "--build", $buildDir,
    "--config", $Config,
    "-j", $Jobs
)

if ($Verbose) {
    $buildArgs += "-v"
}

try {
    & cmake @buildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    Write-Success "Build successful"
} catch {
    Write-Error "Build failed: $_"
    exit 1
}

Write-Host ""

# Run tests
if ($Test) {
    Write-Info "Running tests..."
    Push-Location $buildDir
    try {
        & ctest -C $Config --output-on-failure
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Some tests failed"
        } else {
            Write-Success "All tests passed"
        }
    } catch {
        Write-Error "Test execution failed: $_"
    } finally {
        Pop-Location
    }
    Write-Host ""
}

# Run benchmarks
if ($Benchmark) {
    Write-Info "Running benchmarks..."
    $benchDir = Join-Path $buildDir "benchmarks\$Config"
    if (Test-Path $benchDir) {
        Push-Location $benchDir
        try {
            Get-ChildItem -Filter "*.exe" | ForEach-Object {
                Write-Info "Running $($_.Name)..."
                & $_.FullName
            }
            Write-Success "Benchmarks complete"
        } catch {
            Write-Error "Benchmark execution failed: $_"
        } finally {
            Pop-Location
        }
    } else {
        Write-Warning "Benchmark directory not found"
    }
    Write-Host ""
}

# Summary
Write-Success "Build complete!"
Write-Host ""
Write-Info "Executables:"
$exePath = Join-Path $buildDir "$Config\swiftimpute.exe"
if (Test-Path $exePath) {
    Write-Host "  swiftimpute: $exePath"
} else {
    Write-Warning "  swiftimpute executable not found"
}

Write-Host ""
Write-Info "Next steps:"
Write-Host "  Run tests:       .\build.ps1 -Test"
Write-Host "  Run benchmarks:  .\build.ps1 -Benchmark"
Write-Host "  Clean build:     .\build.ps1 -Clean"
Write-Host "  Debug build:     .\build.ps1 -Config Debug"
Write-Host ""
