Write-Host "========================================="
Write-Host "   Packaging PyQt Desktop Application    "
Write-Host "========================================="

# Clean previous builds
Write-Host "Cleaning previous build directories..."
if (Test-Path -Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path -Path "dist") { Remove-Item -Recurse -Force "dist" }

# Create packaging directory if it doesn't exist
if (-Not (Test-Path -Path "packaging")) {
    New-Item -ItemType Directory -Path "packaging" | Out-Null
}

Write-Host "Running PyInstaller..."
# Ensure you have pyinstaller installed: pip install pyinstaller
pyinstaller packaging/app.spec --clean --noconfirm

Write-Host "Copying assets into distribution folder..."
# If you want to bundle dynamic output data outside the .exe bundle so it can be updated
$distPath = "dist/League_Macro_App"

if (Test-Path -Path $distPath) {
    Write-Host "SUCCESS: Application packaged successfully at $distPath"
} else {
    Write-Host "WARNING: Application packaging may have failed. Please check the logs."
}

Write-Host "========================================="
Write-Host "              Build Complete!            "
Write-Host "========================================="
