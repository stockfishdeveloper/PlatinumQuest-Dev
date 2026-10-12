# Real-map preparation baselines, 2026-10-09

160 completed rounds; four per checkpoint/map/profile. All accumulated scores match final engine scores.
Both source checkpoints retain their original SHA-256. Actions are deterministic with zero forced jumps,
routed goals and normal Super Jump/blast permissions. Assisted: noncompetitive, 8 s rescue. Competitive: no rescue.

Full streams/traces/practice states: `C:\Users\doug\AppData\Local\Temp\pq_checks\phase_r_baseline_verified_20261009`.
Values below are means per round. Points show mean +/- standard error (n=4), not a significance verdict.
Map durations and gem values differ; cross-map averages are descriptive, without par normalization.

## assisted

| Map | Base points +/- SE | Clusters points +/- SE | Base falls | Cluster falls | Base rescues | Cluster rescues |
|---|---:|---:|---:|---:|---:|---:|
| KingOfTheMarble | 139.75 +/- 2.39 | 130.75 +/- 3.42 | 8.50 | 6.00 | 0.00 | 0.00 |
| Sprawl | 10.00 +/- 0.00 | 10.75 +/- 0.75 | 0.00 | 0.00 | 10.00 | 9.25 |
| CubeIsle | 23.50 +/- 7.77 | 28.50 +/- 6.45 | 21.50 | 29.75 | 3.75 | 2.50 |
| Bowl | 7.50 +/- 2.02 | 7.50 +/- 2.18 | 0.00 | 0.00 | 4.75 | 4.75 |
| Duplex | 15.50 +/- 7.93 | 19.75 +/- 11.50 | 13.25 | 9.75 | 6.50 | 5.75 |
| FracturedIslands | 0.00 +/- 0.00 | 64.75 +/- 11.76 | 0.25 | 22.75 | 9.50 | 0.00 |
| Cragmire | 6.75 +/- 2.36 | 5.50 +/- 1.19 | 20.25 | 14.75 | 4.25 | 5.00 |
| VortexEffect | 0.25 +/- 0.25 | 1.00 +/- 0.58 | 116.75 | 81.75 | 0.00 | 0.25 |
| TreasureBox | 2.00 +/- 1.41 | 31.75 +/- 15.43 | 0.00 | 0.50 | 8.25 | 6.25 |
| Concentric | 2.25 +/- 0.25 | 7.00 +/- 3.32 | 30.50 | 21.50 | 3.25 | 5.50 |

Equal-map raw-point mean: base 20.75, clusters 30.73.
Observed map wins: {'base': 2, 'clusters': 7, 'tie': 1}.

## competitive

| Map | Base points +/- SE | Clusters points +/- SE | Base falls | Cluster falls | Base rescues | Cluster rescues |
|---|---:|---:|---:|---:|---:|---:|
| KingOfTheMarble | 148.50 +/- 3.71 | 132.25 +/- 3.01 | 5.50 | 6.50 | 0.00 | 0.00 |
| Sprawl | 4.50 +/- 1.50 | 5.50 +/- 2.06 | 0.00 | 0.00 | 0.00 | 0.00 |
| CubeIsle | 17.25 +/- 2.21 | 7.75 +/- 3.66 | 18.75 | 29.00 | 0.00 | 0.00 |
| Bowl | 3.50 +/- 0.50 | 9.50 +/- 7.54 | 0.00 | 0.00 | 0.00 | 0.00 |
| Duplex | 9.00 +/- 3.19 | 17.50 +/- 2.60 | 2.00 | 4.00 | 0.00 | 0.00 |
| FracturedIslands | 0.25 +/- 0.25 | 69.00 +/- 10.34 | 0.25 | 17.00 | 0.00 | 0.00 |
| Cragmire | 3.75 +/- 1.89 | 7.00 +/- 4.06 | 16.75 | 13.50 | 0.00 | 0.00 |
| VortexEffect | 0.25 +/- 0.25 | 1.00 +/- 0.58 | 117.00 | 95.50 | 0.00 | 0.00 |
| TreasureBox | 0.00 +/- 0.00 | 15.00 +/- 5.40 | 0.00 | 0.00 | 0.00 | 0.00 |
| Concentric | 1.25 +/- 0.75 | 5.75 +/- 3.33 | 1.25 | 5.50 | 0.00 | 0.00 |

Equal-map raw-point mean: base 18.82, clusters 27.02.
Observed map wins: {'base': 2, 'clusters': 8, 'tie': 0}.

## Practice artifacts

{
  "total": 224,
  "dev": 165,
  "eval": 59
}

Practice datasets are prepared but not enabled for training. Preserve TEMP artifacts before cleanup.

## Checkpoint SHA-256

* `nav_points_20261005_parity_32565.pth`: `aacc9545e1c4b4e14f655e249cb6614385ab159bd74d244baccf0c1792634c47`
* `nav_latest.pth`: `167997b13ff6074f4d76e440bba987aad820c8b982bc7a8d5ffd73ca9fb149b0`
