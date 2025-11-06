# Helper script to set environment and build SwiftImpute
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:PATH = "C:\Program Files\CMake\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;" + $env:PATH

Write-Host "Environment configured:"
Write-Host "  CUDA_PATH: $env:CUDA_PATH"
Write-Host "  CMake: $(& cmake --version | Select-Object -First 1)"
Write-Host ""

# Configure
Write-Host "Configuring CMake..."
& cmake -B build `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCUDA_ARCHITECTURES="89" `
    -DCudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Configuration successful!"
    Write-Host ""
    Write-Host "Building..."
    & cmake --build build --config Release -j 8

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Build successful!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Build failed!" -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "Configuration failed!" -ForegroundColor Red
}
