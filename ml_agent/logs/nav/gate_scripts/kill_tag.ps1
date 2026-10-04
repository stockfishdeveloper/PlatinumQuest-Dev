param([string]$Tag, [string]$PortRx)
$procs = Get-CimInstance Win32_Process | Where-Object { ($_.CommandLine -match ("--tag " + $Tag) -or $_.CommandLine -match ("-aiport " + $PortRx)) -and $_.ProcessId -ne $PID -and $_.Name -ne 'bash.exe' }
foreach ($p in $procs) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; "stopped $($p.ProcessId) $($p.Name)" } catch {} }
