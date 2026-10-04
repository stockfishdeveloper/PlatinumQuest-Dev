param([string]$Rx)
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $Rx -and $_.CommandLine -notmatch 'kill_proc' -and $_.ProcessId -ne $PID }
foreach ($p in $procs) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; "stopped $($p.ProcessId) $($p.Name)" } catch {} }
