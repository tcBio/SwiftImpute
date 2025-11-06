# Simple rebuild script
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:PATH = "C:\Program Files\CMake\bin;" + $env:PATH

Write-Host "Building SwiftImpute..." -ForegroundColor Cyan
& cmake --build build --config Release -j 8

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build successful!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Build failed!" -ForegroundColor Red
    Write-Host "Exit code: $LASTEXITCODE"
}
