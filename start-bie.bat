@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  BENNI INFERENCE ENGINE (BIE) -- LOCAL SOVEREIGN RUNTIME
echo  Modelo: Cyber-Tiel-Coder-35B + Gemma-4-26B Vision
echo  Hardware: NVIDIA GeForce RTX 4060 (CUDA Hybrid Offload)
echo ==========================================================

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-bie.ps1" %*
