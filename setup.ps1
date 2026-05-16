# XIM Native Bootstrapper & Installer
# This script installs the Rust toolchain with Cranelift and compiles XIM natively.

$ErrorActionPreference = "Stop"

Write-Host "--- XIM Native Bootstrapper ---" -ForegroundColor Cyan

# 1. Check for Git
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git not found. Installing via WinGet..." -ForegroundColor Yellow
    winget install --id Git.Git -e --source winget
}

# 2. Check for Rustup
if (!(Get-Command rustup -ErrorAction SilentlyContinue)) {
    Write-Host "Rustup not found. Downloading installer..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "rustup-init.exe"
    ./rustup-init.exe -y --default-toolchain nightly --component rustc-codegen-cranelift-preview
    Remove-Item "rustup-init.exe"
    $env:Path += ";$env:USERPROFILE\.cargo\bin"
} else {
    Write-Host "Updating Rust toolchain..." -ForegroundColor Gray
    rustup toolchain install nightly
    rustup component add rustc-codegen-cranelift-preview --toolchain nightly
}

# 3. Clone Repository (if not already in a repo)
if (!(Test-Path ".git")) {
    Write-Host "Cloning XIM Repository..." -ForegroundColor Gray
    git clone https://github.com/turtle170/XIM.git .
}

# 4. Compile XIM with Native Optimization and Cranelift
Write-Host "Compiling XIM with target-cpu=native and Cranelift JIT..." -ForegroundColor Green

# Set RUSTFLAGS for native optimization
$env:RUSTFLAGS = "-C target-cpu=native"

# Build release
cargo +nightly build --release

# 5. Setup Python Extension
if (Test-Path "target/release/xim.dll") {
    Copy-Item "target/release/xim.dll" "xim.pyd" -Force
    Write-Host "XIM Python extension installed (xim.pyd)." -ForegroundColor Green
}

# 6. Finalize
Write-Host "--- XIM Setup Complete ---" -ForegroundColor Cyan
Write-Host "You can now use 'import xim' in Python or run 'xim.exe' from the CLI."
