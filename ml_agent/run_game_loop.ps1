# run_game_loop.ps1 -- keep the game running for unattended training.
#
# Launches marbleblast.exe with "-autotrain <MissionName>"; the hook at the end
# of platinum/client/scripts/ai/mlAgent.cs reads that flag and hosts, loads and
# starts a round of that mission by itself (the engine's own -mission argument
# is inert on the client in this build). The game is relaunched whenever it
# exits, whether it crashed or was closed. The engine prints one harmless
# "Unkown command line argument: -autotrain" line at startup.
#
# Start the trainer first: python train_ppo.py keeps its networks, optimizer
# and counters in memory across game disconnects, so a relaunched game simply
# resumes training at the same update.
#
#   .\run_game_loop.ps1                          # King of the Marble, relaunch on exit only
#   .\run_game_loop.ps1 -RestartEveryHours 6     # also restart proactively every 6 h
#
# Note: if Windows shows a "marbleblast.exe has stopped working" dialog after a
# crash, the process lingers until it is dismissed and this loop waits with it.
# To suppress the dialog: set DWORD DontShowUI = 1 under
#   HKCU\Software\Microsoft\Windows\Windows Error Reporting
# Ctrl+C in this window stops the loop (and leaves the current game running).

param(
    [string]$Mission = "KingOfTheMarble_Hunt",
    [double]$RestartEveryHours = 0
)

$exe = Join-Path $PSScriptRoot "..\Marble Blast Platinum\marbleblast.exe"
$exe = [System.IO.Path]::GetFullPath($exe)
if (-not (Test-Path $exe)) { Write-Error "Game executable not found: $exe"; exit 1 }
$workdir = Split-Path $exe

while ($true) {
    $started = Get-Date
    Write-Host ("[{0}] launching {1} -autotrain {2}" -f $started.ToString("HH:mm:ss"), $exe, $Mission)
    $p = Start-Process -FilePath $exe -ArgumentList @("-autotrain", $Mission) -WorkingDirectory $workdir -PassThru

    if ($RestartEveryHours -gt 0) {
        $deadline = $started.AddHours($RestartEveryHours)
        while (-not $p.HasExited -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 10 }
        if (-not $p.HasExited) {
            Write-Host ("[{0}] proactive restart after {1} h" -f (Get-Date).ToString("HH:mm:ss"), $RestartEveryHours)
            $p.Kill(); $p.WaitForExit()
        }
    } else {
        $p.WaitForExit()
    }

    $hours = ((Get-Date) - $started).TotalHours
    Write-Host ("[{0}] game exited (code {1}) after {2:N2} h; relaunching in 10 s" -f (Get-Date).ToString("HH:mm:ss"), $p.ExitCode, $hours)
    Start-Sleep -Seconds 10
}
