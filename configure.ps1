# Configure script with explicit CUDA paths
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:PATH = "C:\Program Files\CMake\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;" + $env:PATH

# Clean old build
if (Test-Path build) {
    Remove-Item -Recurse -Force build
    Write-Host "Cleaned old build directory"
}

Write-Host "Configuring with CMake..."
Write-Host "CUDA_PATH: $env:CUDA_PATH"

# Try configuration with explicit CUDA settings
& cmake -B build `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -T "cuda=$env:CUDA_PATH" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCUDA_ARCHITECTURES="89"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Configuration successful!" -ForegroundColor Green

    Write-Host ""
    Write-Host "Building..."
    & cmake --build build --config Release -j 8

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Build successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Executable: build\Release\swiftimpute.exe"
    } else {
        Write-Host ""
        Write-Host "Build failed!" -ForegroundColor Red
    }
} else {
    Write-Host "Configuration failed!" -ForegroundColor Red
}
