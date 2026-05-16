# XIM Deployment & CI Trigger Script

Write-Host "--- XIM Deployment Pipeline ---" -ForegroundColor Cyan

# 1. Run Tests
Write-Host "Running tests..."
cargo test
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed! Aborting." -ForegroundColor Red
    exit 1
}

# 2. Format check
Write-Host "Checking formatting..."
cargo fmt --all -- --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "Formatting issues found. Run 'cargo fmt' and try again." -ForegroundColor Yellow
}

# 3. Git Push
Write-Host "Pushing to GitHub (main)..."
git add .
git commit -m "Deployment: Hierarchical JIT Fusion, Cranelift Integration, and Native Bootstrapping"
git push origin main

Write-Host "Deployment script finished." -ForegroundColor Green
