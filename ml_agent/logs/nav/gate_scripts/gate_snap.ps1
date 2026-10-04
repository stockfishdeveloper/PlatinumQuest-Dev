param([string]$Ckpt, [string]$Tag, [int]$Port0)
$ml = "C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent"
Set-Location $ml
$env:NAV_CKPT = $Ckpt
& "$ml\nav\learned_nav\run_many.ps1" -Module nav.learned_nav.hybrid -Map KingOfTheMarble_Hunt -N 4 -Port0 $Port0 -Tag $Tag -ArgStr "--rounds 2 --memory current --shortcuts 1 --rescue 1"
