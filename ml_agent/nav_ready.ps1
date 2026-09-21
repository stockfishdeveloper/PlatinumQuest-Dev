# nav_ready.ps1 -- dot-source this for the READY-AT-GO helpers.
#
# THE RULE (user, 2026-09-21, no exceptions): every time the game is run with the intent to roll
# the marble, everything must be ready to roll the instant "GO!" appears. No warm-up, no dead
# marble on the start pad.
#
# WHY IT WAS BROKEN. Python is the socket SERVER (nav/env.py:51-53 bind + listen) and the game is
# the CLIENT: AIBridge::connect dials us 100 ms after GO (mlAgent.cs:677-686). Every launcher used
# to start the GAME first and then sleep 20-30 s before starting Python, so the game counted down,
# said GO, found nothing listening, and the marble sat there until Python bound its port.
#
# THE ORDER, everywhere: start Python -> Wait-NavPort (a positive check that the port is actually
# listening, never a blind sleep) -> launch the game. The game is then answered on its first dial.
#
#   . "$PSScriptRoot\nav_ready.ps1"
#   Wait-NavPort -Port 8920 -TimeoutSec 120
#   Wait-NavPorts -Port0 8888 -Count 8

function Wait-NavPort {
    <#  Block until something is LISTENING on $Port. Returns $true when it is, $false on timeout. #>
    param([int]$Port, [int]$TimeoutSec = 120, [switch]$Quiet)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $up = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
        if ($up.Count -gt 0) {
            if (-not $Quiet) { Write-Host ("[{0}] port {1} is listening; safe to launch the game" -f (Get-Date).ToString("HH:mm:ss"), $Port) }
            return $true
        }
        Start-Sleep -Milliseconds 200
    }
    Write-Warning ("port {0} never opened within {1}s: the game would reach GO with nobody listening" -f $Port, $TimeoutSec)
    return $false
}

function Wait-NavPorts {
    <#  Block until ALL of $Count consecutive ports from $Port0 are listening (the 8 training
        workers). Returns $true only if every one of them came up.  #>
    param([int]$Port0, [int]$Count, [int]$TimeoutSec = 180, [switch]$Quiet)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $want = @($Port0..($Port0 + $Count - 1))
    while ((Get-Date) -lt $deadline) {
        $up = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $want -contains $_.LocalPort } | Select-Object -ExpandProperty LocalPort -Unique)
        if ($up.Count -ge $Count) {
            if (-not $Quiet) { Write-Host ("[{0}] all {1} worker ports {2}..{3} listening; safe to launch the games" -f (Get-Date).ToString("HH:mm:ss"), $Count, $Port0, ($Port0 + $Count - 1)) }
            return $true
        }
        Start-Sleep -Milliseconds 250
    }
    $up = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $want -contains $_.LocalPort } | Select-Object -ExpandProperty LocalPort -Unique)
    Write-Warning ("only {0}/{1} worker ports listening after {2}s" -f $up.Count, $Count, $TimeoutSec)
    return $false
}
