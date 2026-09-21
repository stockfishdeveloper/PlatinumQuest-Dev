# run_game_loop.ps1 -- keep the game instances running for unattended training.
#
# Launches N copies of the game with "-autotrain <MissionName> -aiport <port>"; the hook at the
# end of platinum/client/scripts/ai/mlAgent.cs reads those flags and hosts, loads and starts a
# round of that mission by itself, connecting to the trainer on its port (8888, 8889, ...).
# Any instance that exits (crash or close) is relaunched on the same port with the same mission.
# The engine prints one harmless "Unkown command line argument: -autotrain" line at startup.
#
# Start the trainer first: python -m nav.train_nav keeps its networks, optimizer and counters in
# memory across game disconnects, so a relaunched instance simply resumes on its port.
#
#   .\run_game_loop.ps1                                             # 16 instances, all FlatIslands_Hunt
#   .\run_game_loop.ps1 -Mission "FlatIslands_Hunt,KingOfTheMarble_Hunt" -Split "8,8"
#                                                                   # instances 0-7 islands, 8-15 KOTM (a map mix,
#                                                                   #  every update sees both maps; no forgetting)
#   .\run_game_loop.ps1 -Mission "A,B,C"                            # no -Split: round-robin A,B,C,A,B,C,...
#   .\run_game_loop.ps1 -Instances 1 -Exe marbleblast.exe           # the shipped exe (one instance only: its mutex)
#   .\run_game_loop.ps1 -RestartEveryHours 6                        # also relaunch everything every 6 h
#
# The built engine (marbleblast_mbx.exe, OpenPQ-TGEMIT branch ai-training-mode) runs several
# instances at once and in fixed-step lockstep does not care whether its windows are focused.
# The shipped exe needs its window in the foreground (rtf 1.5 otherwise), which is done below.
#
# Note: if Windows shows a "marbleblast.exe has stopped working" dialog after a crash, the
# process lingers until it is dismissed. To suppress the dialog: set DWORD DontShowUI = 1 under
#   HKCU\Software\Microsoft\Windows\Windows Error Reporting
# Ctrl+C in this window stops the loop (and leaves the current games running).

param(
    [string]$Mission = "FlatIslands_Hunt",         # one mission, or a comma-separated list spread over the instances
    [string]$Split = "",                           # how many instances per mission, e.g. "6,2" (default: round-robin)
    [double]$RestartEveryHours = 0,                # relaunch every instance this often (0 = only when one exits)
    [int]$Instances = 8,                           # must match N_INSTANCES in nav/train_nav.py
    [string]$Exe = "marbleblast_mbx.exe",          # built engine; marbleblast.exe = the shipped one
    [int]$Port0 = 8888
)
$missions = @($Mission -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })   # @() keeps a single mission an array (a scalar would index its first LETTER)

# mission per instance index
$assign = @()
if ($Split -ne "") {
    $counts = @($Split -split "," | ForEach-Object { [int]$_.Trim() })
    if ($counts.Count -ne $missions.Count) { Write-Error "-Split needs one count per mission ($($missions.Count) missions, $($counts.Count) counts)"; exit 1 }
    for ($m = 0; $m -lt $missions.Count; $m++) { for ($k = 0; $k -lt $counts[$m]; $k++) { $assign += $missions[$m] } }
    if ($assign.Count -ne $Instances) { Write-Error "-Split adds up to $($assign.Count) instances, -Instances is $Instances"; exit 1 }
} else {
    for ($i = 0; $i -lt $Instances; $i++) { $assign += $missions[$i % $missions.Count] }
}

$exe = Join-Path $PSScriptRoot "..\Marble Blast Platinum\$Exe"
$exe = [System.IO.Path]::GetFullPath($exe)
if (-not (Test-Path $exe)) { Write-Error "Game executable not found: $exe"; exit 1 }
$workdir = Split-Path $exe

function Launch-Instance([int]$i) {
    $port = $Port0 + $i
    $mission = $assign[$i]
    $p = Start-Process -FilePath $exe -ArgumentList @("-autotrain", $mission, "-aiport", "$port") -WorkingDirectory $workdir -PassThru
    Write-Host ("[{0}] instance {1} launched: {2} -autotrain {3} -aiport {4} (pid {5})" -f (Get-Date).ToString("HH:mm:ss"), $i, $Exe, $mission, $port, $p.Id)
    return $p
}

while ($true) {
    $started = Get-Date
    Write-Host ("[{0}] mission per instance: {1}" -f $started.ToString("HH:mm:ss"), ($assign -join ", "))
    $procs = @()
    for ($i = 0; $i -lt $Instances; $i++) {
        $procs += Launch-Instance $i
        Start-Sleep -Seconds 2
    }

    if ($Exe -eq "marbleblast.exe") {
        # Shipped exe: bring the window to the foreground once it is up (behind other windows the
        # engine runs at half speed). Plain SetForegroundWindow is refused for a background
        # caller; an ALT keypress followed by AppActivate is not.
        Start-Sleep -Seconds 25
        try {
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.SendKeys]::SendWait('%')
            (New-Object -ComObject WScript.Shell).AppActivate($procs[0].Id) | Out-Null
        } catch { Write-Host "could not activate the game window: $($_.Exception.Message)" }
    }

    $deadline = if ($RestartEveryHours -gt 0) { $started.AddHours($RestartEveryHours) } else { [DateTime]::MaxValue }
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        for ($i = 0; $i -lt $Instances; $i++) {
            if ($procs[$i].HasExited) {
                Write-Host ("[{0}] instance {1} exited (code {2}); relaunching in 10 s" -f (Get-Date).ToString("HH:mm:ss"), $i, $procs[$i].ExitCode)
                Start-Sleep -Seconds 10
                $procs[$i] = Launch-Instance $i
            }
        }
    }
    Write-Host ("[{0}] proactive restart after {1} h: closing {2} instance(s)" -f (Get-Date).ToString("HH:mm:ss"), $RestartEveryHours, $Instances)
    foreach ($p in $procs) { if (-not $p.HasExited) { $p.Kill(); $p.WaitForExit() } }
    Start-Sleep -Seconds 5
}
