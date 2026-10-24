param([string]$Tag, [string]$Seeds = "7347019,12091843", [string]$Flag = "--edge-hits")
$ml = "C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent"; $root = "C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev"
$env:PYTHONPATH = $ml; $py = "C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe"
$maps = @('KingOfTheMarble_Hunt','Sprawl_Hunt','Bowl_Hunt','CubeIsle_Hunt','Duplex_Hunt','Cragmire_Hunt','FracturedIslands_Hunt','VortexEffect_Hunt','TreasureBox_Hunt','Concentric_Hunt')
$seedList = $Seeds.Split(',')
$common = @('-u','-m','nav.marble_dynamics.reliability_eval','--maps') + $maps + @('--seeds') + $seedList + @('--random-starts','--only-branch','physics_plan','--full-round','--horizon','48','--commit','4','--reuse-plan','--geometry-route','--resource-acquisition','--recharge-continuation','--landing-search','--proposal-samples','20','--query-cache','--clearance-cache','--global-goals','--physics-frame-ms','16','--fallback-cooldown','2','--search-budget-s','2','--pickup-landing-continuation','40','--component-crossing','--ballistic-landing','--use-horizon','48','--use-margin-s','1','--cache-dir','C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013','--sweep-dll','C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll','--workers','4')
$out = "$ml\dynamics_reports\handoff_safety_20261020"; $logd = "$env:TEMP\pq_checks"
$b = Start-Process -FilePath $py -ArgumentList ($common + @('--port','9100','--output',"$out\${Tag}_baseline")) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$logd\${Tag}_baseline.log" -RedirectStandardError "$logd\${Tag}_baseline.err" -PassThru
$candArgs = $common + @('--port','9104','--output',"$out\${Tag}_candidate")
if ($Flag -ne "") { $candArgs = $common + @($Flag.Split(" ")) + @('--port','9104','--output',"$out\${Tag}_candidate") }
$c = Start-Process -FilePath $py -ArgumentList $candArgs -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$logd\${Tag}_candidate.log" -RedirectStandardError "$logd\${Tag}_candidate.err" -PassThru
Start-Sleep 50
"baseline PID $($b.Id) exited=$($b.HasExited)  candidate PID $($c.Id) exited=$($c.HasExited)  $(Get-Date -Format HH:mm:ss) games=" + (Get-Process marbleblast* -ErrorAction SilentlyContinue | Measure-Object).Count
