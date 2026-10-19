#!/bin/bash
# usage: sim_batch.sh <outdir> <parallel> <common args...> -- <variant spec> [<variant spec> ...]
# variant spec: name:extra args separated by commas, e.g. "align:--align,1"
S="$(cd "$(dirname "$0")" && pwd)"
out=$1; par=$2; shift 2; common=()
while [ "$1" != "--" ]; do common+=("$1"); shift; done; shift
MAPS=${MAPS:-"KingOfTheMarble_Hunt Sprawl_Hunt CubeIsle_Hunt Bowl_Hunt Duplex_Hunt FracturedIslands_Hunt Cragmire_Hunt VortexEffect_Hunt TreasureBox_Hunt Concentric_Hunt"}
mkdir -p "$out"; jobs="$out/jobs.txt"; : > "$jobs"
if [ -n "$VARIANT_MAJOR" ]; then for spec in "$@"; do for m in $MAPS; do echo "$m $spec" >> "$jobs"; done; done
else for m in $MAPS; do for spec in "$@"; do echo "$m $spec" >> "$jobs"; done; done; fi
cd "C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev"
export PYTHONPATH=ml_agent
cat "$jobs" | xargs -P "$par" -L 1 bash -c 'm=$0; spec=$1; v=${spec%%:*}; extra=${spec#*:}; [ "$extra" = "$spec" ] && extra=""; python "'"$S"'/sim_bench.py" --map $m --variant $v ${extra//,/ } '"${common[*]}"' --out "'"$out"'/$v.$m.jsonl" > "'"$out"'/$v.$m.log" 2>&1; echo "job done $v $m $(grep -c . "'"$out"'/$v.$m.jsonl" 2>/dev/null)"'
echo "BATCH DONE $(date +%H:%M)"
