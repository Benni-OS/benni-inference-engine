param(
    [int]$Port = 8088,
    [string]$HostAddress = "127.0.0.1",
    [int]$Context = 8192,
    [switch]$Background
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPython = Join-Path $ScriptDir ".venv-bie-runtime\Scripts\python.exe"
$ModelFile = Join-Path $ScriptDir "bie\models\Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERRO: Ambiente virtual nao encontrado: $VenvPython" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $ModelFile)) {
    Write-Host "ERRO: Modelo nao encontrado: $ModelFile" -ForegroundColor Red
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host " BENNI INFERENCE ENGINE (BIE) -- LOCAL SOVEREIGN RUNTIME" -ForegroundColor Cyan
Write-Host " Modelo: Cyber-Tiel-Coder-35B (MoE 35B / 3.5B ativo)" -ForegroundColor Green
Write-Host " Hardware: NVIDIA GeForce RTX 4060 (CUDA Hybrid Offload)" -ForegroundColor Blue
Write-Host " Endpoint OpenAI: http://$HostAddress`:$Port/v1" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Magenta

# Checa se ja esta ativo
try {
    $r = Invoke-WebRequest -Uri "http://$HostAddress`:$Port/health" -Method Get -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
    if ($r -and $r.StatusCode -eq 200) {
        Write-Host "BIE ja esta ativo e respondendo na porta $Port!" -ForegroundColor Green
        exit 0
    }
} catch {
}

$LogDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$LogOut = Join-Path $LogDir "bie-$Port-stdout.log"
$LogErr = Join-Path $LogDir "bie-$Port-stderr.log"

$env:PYTHONPATH = $ScriptDir

if ($Background) {
    Write-Host "Iniciando BIE em segundo plano na porta $Port..." -ForegroundColor Yellow
    $cmdLine = "`"$VenvPython`" -m bie.serve serve --host $HostAddress --port $Port --context $Context --model `"$ModelFile`" > `"$LogOut`" 2> `"$LogErr`""
    $spawned = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd.exe /s /c `"$cmdLine`""
        CurrentDirectory = $ScriptDir
    }
    if ($spawned.ReturnValue -ne 0 -or -not $spawned.ProcessId) {
        Write-Host "ERRO: Falha ao iniciar processo desacoplado BIE (codigo: $($spawned.ReturnValue))." -ForegroundColor Red
        exit 1
    }
    $wrapperPid = [int]$spawned.ProcessId

    Write-Host "Processo BIE iniciado (PID: $wrapperPid). Carregando pesos na GPU..." -ForegroundColor Cyan

    $ready = $false
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 1
        try {
            $check = Invoke-WebRequest -Uri "http://$HostAddress`:$Port/health" -Method Get -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($check -and $check.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
        }
        $alive = Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue
        if (-not $alive) {
            Write-Host "ERRO: Processo BIE encerrou inesperadamente. Verifique $LogErr" -ForegroundColor Red
            if (Test-Path $LogErr) { Get-Content $LogErr -Tail 20 }
            exit 1
        }
        Write-Host "." -NoNewline -ForegroundColor DarkGray
    }
    Write-Host ""

    if ($ready) {
        Write-Host "Sucesso: BIE online e respondendo em http://$HostAddress`:$Port/v1" -ForegroundColor Green
    } else {
        Write-Host "Aviso: O modelo ainda esta carregando. Monitore http://$HostAddress`:$Port/health" -ForegroundColor Yellow
    }
} else {
    Write-Host "Iniciando BIE em primeiro plano..." -ForegroundColor Yellow
    & $VenvPython -m bie.serve serve --host $HostAddress --port $Port --context "$Context" --model "$ModelFile"
}
