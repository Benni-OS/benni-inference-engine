# Encerra processos do BIE caso estejam rodando
param([int]$Port = 8088)

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($connections) {
    foreach ($conn in $connections) {
        $pidToKill = $conn.OwningProcess
        if ($pidToKill -gt 0) {
            Write-Host "Encerrando processo $pidToKill na porta $Port..." -ForegroundColor Yellow
            Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "Porta $Port liberada." -ForegroundColor Green
} else {
    Write-Host "Nenhum processo BIE ativo na porta $Port." -ForegroundColor Cyan
}
