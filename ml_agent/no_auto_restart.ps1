# Stop the PC restarting on its own (operator, 2026-10-08, after an unexpected reboot killed a training run).
# Run ELEVATED:  powershell -ExecutionPolicy Bypass -File .\no_auto_restart.ps1
# Everything here is reversible: see the notes at the end of each block.
$ErrorActionPreference = 'Continue'
function Say($m) { Write-Host ("[{0}] {1}" -f (Get-Date).ToString('HH:mm:ss'), $m) }
$elev = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elev) { Write-Error 'run this elevated (right-click PowerShell, Run as administrator)'; exit 1 }

# 1. Windows Update: no automatic download/install, never reboot while someone is logged on.
$au = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU'
New-Item -Path $au -Force | Out-Null
Set-ItemProperty -Path $au -Name NoAutoUpdate -Type DWord -Value 1                      # updates only when you click Check for updates
Set-ItemProperty -Path $au -Name AUOptions -Type DWord -Value 2                         # 2 = notify before download (belt and braces)
Set-ItemProperty -Path $au -Name NoAutoRebootWithLoggedOnUsers -Type DWord -Value 1     # never auto-restart with a user logged on
Set-ItemProperty -Path $au -Name AlwaysAutoRebootAtScheduledTime -Type DWord -Value 0
Say 'Windows Update policy: NoAutoUpdate=1, AUOptions=2, NoAutoRebootWithLoggedOnUsers=1'
# undo: Remove-Item $au -Recurse

# 2. Also pause updates for the maximum 35 days (the Home edition honours this even where it ignores policies).
$ux = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
New-Item -Path $ux -Force | Out-Null
$until = (Get-Date).AddDays(35).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$start = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
foreach ($n in 'PauseUpdatesExpiryTime','PauseFeatureUpdatesEndTime','PauseQualityUpdatesEndTime') { Set-ItemProperty -Path $ux -Name $n -Type String -Value $until }
foreach ($n in 'PauseFeatureUpdatesStartTime','PauseQualityUpdatesStartTime') { Set-ItemProperty -Path $ux -Name $n -Type String -Value $start }
Set-ItemProperty -Path $ux -Name PauseUpdatesStartTime -Type String -Value $start
# active hours: the widest window the Home edition allows (18 h), 06:00-24:00
Set-ItemProperty -Path $ux -Name ActiveHoursStart -Type DWord -Value 6
Set-ItemProperty -Path $ux -Name ActiveHoursEnd -Type DWord -Value 0
Set-ItemProperty -Path $ux -Name IsActiveHoursEnabled -Type DWord -Value 1
Say "updates paused until $until; active hours 06:00-24:00"
# undo: Settings > Windows Update > Resume updates

# 3. The reboot tasks of the update orchestrator (best effort: some are protected and will say Access denied).
foreach ($t in 'Reboot_AC','Reboot_Battery','USO_UxBroker','Schedule Scan','Schedule Scan Static Task','UpdateModelTask','Start Oobe Expedite Work') {
    $r = schtasks /Change /TN "\Microsoft\Windows\UpdateOrchestrator\$t" /DISABLE 2>&1
    Say ("task {0}: {1}" -f $t, ($r | Select-Object -Last 1))
}
# undo: schtasks /Change /TN "\Microsoft\Windows\UpdateOrchestrator\<name>" /ENABLE

# 4. No automatic restart after a system crash: a blue screen stays on screen so the cause can be read.
Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' -Name AutoReboot -Type DWord -Value 0
Say 'crash control: AutoReboot=0 (a BSOD stays up instead of rebooting)'
# undo: set AutoReboot to 1

# 5. Never sleep or hibernate on AC; keep hybrid sleep and wake timers off so nothing wakes/ sleeps the box mid-run.
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 0
powercfg /setactive SCHEME_CURRENT
Say 'power: no sleep/hibernate/monitor-off on AC, wake timers off'
# undo: powercfg /change standby-timeout-ac 30  (etc.)

# 6. Kernel-power 41 was the symptom today: a hard crash/hang/power loss, not an update. Make sure a dump is kept.
Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' -Name CrashDumpEnabled -Type DWord -Value 7   # 7 = automatic memory dump
Say 'crash dumps: automatic memory dump on'
Say 'DONE. Settings > Windows Update will show updates paused; a restart is never scheduled while you are logged on.'
