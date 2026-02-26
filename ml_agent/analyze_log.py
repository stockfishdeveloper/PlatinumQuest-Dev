"""
Training Log Analyzer for PlatinumQuest PPO

Parses training log files and prints a concise summary with key metrics,
trends, and problem detection.

Usage:
    python analyze_log.py                    # Analyze most recent log
    python analyze_log.py logs/training_*.log  # Analyze specific log
    python analyze_log.py --last 50          # Last 50 episodes only
"""

import re
import sys
import os
import glob
import argparse
from collections import defaultdict


def parse_log(filepath):
    """Parse a training log file into structured data."""
    updates = []
    episodes = []
    summaries = []

    # Regex patterns
    upd_re = re.compile(
        r'Upd\s+(\d+)\s*\|\s*'
        r'PL=([-\d.]+)\s+VL=([-\d.]+)\s+'
        r'Ent=([-\d.]+)\s+GN=([-\d.]+)\s+'
        r'KL=([-\d.]+)\s*\|\s*'
        r'AvgRwd=([-\d.]+)\s+AvgLen=(\d+)'
        r'(.*?)$'
    )
    ep_re = re.compile(
        r'Ep\s+(\d+)\s+\[(\w+)\]\s+'
        r'rwd=([-\d.]+)\s+gems=(\d+)pts\s+'
        r'OOB=(\d+)\s+steps=(\d+)\s*\|\s*'
        r'avg100=([-\d.]+)'
    )
    gem_re = re.compile(r'\[GEM\].*?\+(\d+)pts')
    summary_re = re.compile(
        r'SUMMARY Upd (\d+)\s*\|\s*([\d.]+)h\s*\|\s*([\d,]+)\s*steps\s*\|\s*(\d+)\s*eps'
    )
    summary_gems_re = re.compile(
        r'Gems:\s*(\d+)pts\s*\(([\d.]+)/hr\)\s*\|\s*OOB:\s*(\d+)\s*\|\s*'
        r'AvgRwd:\s*([-\d.]+)\s*\|\s*Best:\s*([-\d.]+)'
    )
    summary_detail_re = re.compile(
        r'AvgEpLen:\s*(\d+)\s*steps\s*\|\s*Ent:\s*([-\d.]+)\s*\|\s*'
        r'GN:\s*([-\d.]+)\s*\|\s*Lazy:\s*([-\d.]+)\s*\|\s*Std:\s*([-\d.]+)'
    )
    policystd_re = re.compile(r'PolicyStd:([\d.]+)')
    load_re = re.compile(r'Loading model from (.+)')
    init_std_re = re.compile(r'PolicyStd:\s*([\d.]+)\s*degrees\s*->\s*annealing to\s*([\d.]+)')

    load_info = {}

    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()

            # Load info
            m = load_re.search(line)
            if m:
                load_info['checkpoint'] = m.group(1)

            m = init_std_re.search(line)
            if m:
                load_info['init_std'] = float(m.group(1))
                load_info['target_std'] = float(m.group(2))

            # Update lines
            m = upd_re.search(line)
            if m:
                tail = m.group(9)
                gems_m = re.search(r'gems=(\d+)', tail)
                oob_m = re.search(r'OOB=(\d+)', tail)
                dry_m = re.search(r'DRY.(\d+)', tail)
                pstd_m = policystd_re.search(line)
                kl_stopped = 'KL-STOP' in tail

                updates.append({
                    'update': int(m.group(1)),
                    'policy_loss': float(m.group(2)),
                    'value_loss': float(m.group(3)),
                    'entropy': float(m.group(4)),
                    'grad_norm': float(m.group(5)),
                    'kl': float(m.group(6)),
                    'avg_reward': float(m.group(7)),
                    'avg_len': int(m.group(8)),
                    'gems': int(gems_m.group(1)) if gems_m else 0,
                    'oob': int(oob_m.group(1)) if oob_m else 0,
                    'dry': int(dry_m.group(1)) if dry_m else 0,
                    'policy_std': float(pstd_m.group(1)) if pstd_m else None,
                    'collapse': '*** ENTROPY COLLAPSE ***' in tail,
                    'entropy_low': '(entropy low)' in tail,
                    'kl_stopped': kl_stopped,
                })

            # Episode lines
            m = ep_re.search(line)
            if m:
                episodes.append({
                    'episode': int(m.group(1)),
                    'status': m.group(2),
                    'reward': float(m.group(3)),
                    'gems': int(m.group(4)),
                    'oob': int(m.group(5)),
                    'steps': int(m.group(6)),
                    'avg100': float(m.group(7)),
                })

            # Summary lines
            m = summary_re.search(line)
            if m:
                s = {
                    'update': int(m.group(1)),
                    'hours': float(m.group(2)),
                    'steps': int(m.group(3).replace(',', '')),
                    'episodes': int(m.group(4)),
                }
                summaries.append(s)

            # Summary gem line (attach to last summary)
            m = summary_gems_re.search(line)
            if m and summaries:
                summaries[-1]['total_gems'] = int(m.group(1))
                summaries[-1]['gems_hr'] = float(m.group(2))
                summaries[-1]['total_oob'] = int(m.group(3))
                summaries[-1]['avg_reward'] = float(m.group(4))
                summaries[-1]['best_reward'] = float(m.group(5))

            # Summary detail line
            m = summary_detail_re.search(line)
            if m and summaries:
                summaries[-1]['entropy'] = float(m.group(2))
                summaries[-1]['laziness'] = float(m.group(4))
                summaries[-1]['policy_std'] = float(m.group(5))

    return {
        'updates': updates,
        'episodes': episodes,
        'summaries': summaries,
        'load_info': load_info,
        'filepath': filepath,
    }


def print_analysis(data, last_n=None):
    """Print a comprehensive analysis of parsed log data."""
    updates = data['updates']
    episodes = data['episodes']
    summaries = data['summaries']
    load_info = data['load_info']

    if not updates:
        print("No update data found in log.")
        return

    print("=" * 70)
    print(f"LOG: {os.path.basename(data['filepath'])}")
    print("=" * 70)

    # Load info
    if load_info:
        print(f"Checkpoint: {load_info.get('checkpoint', '?')}")
        if 'init_std' in load_info:
            print(f"Init PolicyStd: {load_info['init_std']:.1f} deg -> target {load_info['target_std']:.1f} deg")

    # Overall stats
    first_upd = updates[0]['update']
    last_upd = updates[-1]['update']
    total_updates = last_upd - first_upd + 1
    print(f"\nUpdates: {first_upd} -> {last_upd} ({total_updates} total)")
    print(f"Episodes: {len(episodes)}")

    if summaries:
        s = summaries[-1]
        hrs = s.get('hours', 0)
        print(f"Duration: {hrs:.2f}h")
        print(f"Total gems: {s.get('total_gems', '?')}pts ({s.get('gems_hr', 0):.0f}/hr)")
        print(f"Total OOB: {s.get('total_oob', '?')}")

    # Slice for analysis
    if last_n and len(episodes) > last_n:
        ep_slice = episodes[-last_n:]
        print(f"\n--- Showing last {last_n} episodes ---")
    else:
        ep_slice = episodes

    # =========================================================================
    # Episode Analysis
    # =========================================================================
    if ep_slice:
        print("\n" + "-" * 70)
        print("EPISODE ANALYSIS")
        print("-" * 70)

        rewards = [e['reward'] for e in ep_slice]
        gems = [e['gems'] for e in ep_slice]
        oobs = [e['oob'] for e in ep_slice]

        print(f"  Reward:  min={min(rewards):.0f}  max={max(rewards):.0f}  "
              f"avg={sum(rewards)/len(rewards):.0f}  median={sorted(rewards)[len(rewards)//2]:.0f}")
        print(f"  Gems:    min={min(gems)}  max={max(gems)}  "
              f"avg={sum(gems)/len(gems):.1f}  median={sorted(gems)[len(gems)//2]}")
        print(f"  OOB:     min={min(oobs)}  max={max(oobs)}  "
              f"avg={sum(oobs)/len(oobs):.1f}  total={sum(oobs)}")

        # Best / worst episodes
        best = max(ep_slice, key=lambda e: e['reward'])
        worst = min(ep_slice, key=lambda e: e['reward'])
        print(f"\n  Best:  Ep {best['episode']}  rwd={best['reward']:.0f}  gems={best['gems']}  OOB={best['oob']}")
        print(f"  Worst: Ep {worst['episode']}  rwd={worst['reward']:.0f}  gems={worst['gems']}  OOB={worst['oob']}")

        # Trend: split into halves
        half = len(ep_slice) // 2
        if half >= 3:
            first_half = ep_slice[:half]
            second_half = ep_slice[half:]
            r1 = sum(e['reward'] for e in first_half) / len(first_half)
            r2 = sum(e['reward'] for e in second_half) / len(second_half)
            g1 = sum(e['gems'] for e in first_half) / len(first_half)
            g2 = sum(e['gems'] for e in second_half) / len(second_half)
            o1 = sum(e['oob'] for e in first_half) / len(first_half)
            o2 = sum(e['oob'] for e in second_half) / len(second_half)

            def arrow(a, b, higher_good=True):
                diff = b - a
                pct = (diff / abs(a) * 100) if a != 0 else 0
                if abs(pct) < 2:
                    return "flat"
                direction = "UP" if diff > 0 else "DOWN"
                good = (diff > 0) == higher_good
                marker = "OK" if good else "BAD"
                return f"{direction} {abs(pct):.0f}% [{marker}]"

            print(f"\n  TREND (1st half -> 2nd half):")
            print(f"    Reward: {r1:.0f} -> {r2:.0f}  {arrow(r1, r2, True)}")
            print(f"    Gems:   {g1:.1f} -> {g2:.1f}  {arrow(g1, g2, True)}")
            print(f"    OOB:    {o1:.1f} -> {o2:.1f}  {arrow(o1, o2, False)}")

    # =========================================================================
    # Training Metrics Trend
    # =========================================================================
    print("\n" + "-" * 70)
    print("TRAINING METRICS PROGRESSION")
    print("-" * 70)

    # Sample at intervals
    n_samples = min(10, len(updates))
    step = max(1, len(updates) // n_samples)
    sample_indices = list(range(0, len(updates), step))
    if sample_indices[-1] != len(updates) - 1:
        sample_indices.append(len(updates) - 1)

    print(f"  {'Upd':>6} {'Ent':>7} {'PStd':>6} {'PL':>8} {'VL':>8} {'GN':>7} {'KL':>7} {'AvgRwd':>9} {'OOB':>4} {'Gems':>5}")
    for i in sample_indices:
        u = updates[i]
        pstd = f"{u['policy_std']:.1f}" if u['policy_std'] else "?"
        kl_str = f"{u['kl']:.4f}" if 'kl' in u else "?"
        kl_flag = "*" if u.get('kl_stopped') else ""
        print(f"  {u['update']:>6} {u['entropy']:>7.3f} {pstd:>6} "
              f"{u['policy_loss']:>8.4f} {u['value_loss']:>8.4f} {u['grad_norm']:>7.3f} "
              f"{kl_str:>7}{kl_flag} "
              f"{u['avg_reward']:>9.1f} {u['oob']:>4} {u['gems']:>5}")

    # =========================================================================
    # Problem Detection
    # =========================================================================
    print("\n" + "-" * 70)
    print("PROBLEM DETECTION")
    print("-" * 70)

    problems = []

    # Entropy collapse
    ent_values = [u['entropy'] for u in updates]
    if ent_values[-1] < -0.5:
        problems.append(f"ENTROPY COLLAPSED: {ent_values[-1]:.3f} (< -0.5)")
    elif ent_values[-1] < 0.3:
        problems.append(f"ENTROPY LOW: {ent_values[-1]:.3f} (< 0.3)")

    # Entropy trend
    if len(ent_values) >= 20:
        ent_start = sum(ent_values[:10]) / 10
        ent_end = sum(ent_values[-10:]) / 10
        ent_drop = ent_start - ent_end
        if ent_drop > 0.5:
            problems.append(f"ENTROPY DROPPING FAST: {ent_start:.3f} -> {ent_end:.3f} (delta={ent_drop:.3f})")

    # OOB trend
    if episodes:
        recent_eps = episodes[-10:] if len(episodes) >= 10 else episodes
        avg_oob = sum(e['oob'] for e in recent_eps) / len(recent_eps)
        if avg_oob > 15:
            problems.append(f"HIGH OOB: avg {avg_oob:.1f}/episode in last {len(recent_eps)} eps")
        elif avg_oob > 8:
            problems.append(f"ELEVATED OOB: avg {avg_oob:.1f}/episode in last {len(recent_eps)} eps")

    # OOB spike detection
    if len(episodes) >= 5:
        for i in range(4, len(episodes)):
            if episodes[i]['oob'] > 3 * max(1, sum(e['oob'] for e in episodes[max(0,i-5):i]) / 5):
                problems.append(f"OOB SPIKE: Ep {episodes[i]['episode']} had {episodes[i]['oob']} OOBs")

    # Reward collapse
    if episodes and len(episodes) >= 5:
        recent = episodes[-5:]
        best_ever = max(e['avg100'] for e in episodes)
        current = recent[-1]['avg100']
        drop_pct = (best_ever - current) / best_ever * 100 if best_ever > 0 else 0
        if drop_pct > 10:
            problems.append(f"REWARD DECLINE: peak avg100={best_ever:.0f} -> current={current:.0f} ({drop_pct:.0f}% drop)")

    # KL early stopping
    kl_stops = [u for u in updates if u.get('kl_stopped')]
    if kl_stops:
        kl_pct = len(kl_stops) / len(updates) * 100
        max_kl_val = max(u.get('kl', 0) for u in updates)
        if kl_pct > 50:
            problems.append(f"KL-STOP EXCESSIVE: {len(kl_stops)}/{len(updates)} updates ({kl_pct:.0f}%) - threshold too tight?")
        elif kl_pct > 10:
            problems.append(f"KL-STOP FREQUENT: {len(kl_stops)} updates ({kl_pct:.0f}%), max KL={max_kl_val:.4f}")
        else:
            problems.append(f"KL-STOP: {len(kl_stops)} updates caught (max KL={max_kl_val:.4f})")

    # KL spikes (even if not stopped)
    kl_values = [u.get('kl', 0) for u in updates]
    kl_spikes = [u for u in updates if u.get('kl', 0) > 1.0]
    if kl_spikes:
        problems.append(f"KL SPIKES: {len(kl_spikes)} updates with KL>1.0 (max={max(u['kl'] for u in kl_spikes):.4f})")

    # Gradient norm spikes
    gn_values = [u['grad_norm'] for u in updates]
    gn_spikes = [u for u in updates if u['grad_norm'] > 50]
    if gn_spikes:
        problems.append(f"GRAD NORM SPIKES: {len(gn_spikes)} updates with GN>50 (max={max(u['grad_norm'] for u in gn_spikes):.1f})")

    # Dry rollout streaks
    max_dry = max(u['dry'] for u in updates) if updates else 0
    if max_dry >= 5:
        problems.append(f"DRY STREAK: {max_dry} consecutive rollouts with 0 gems")

    # PolicyStd
    pstd_values = [u['policy_std'] for u in updates if u['policy_std'] is not None]
    if pstd_values:
        if pstd_values[-1] < 5:
            problems.append(f"POLICY STD VERY LOW: {pstd_values[-1]:.1f} deg (may be too deterministic)")
        if len(pstd_values) >= 10:
            pstd_start = sum(pstd_values[:5]) / 5
            pstd_end = sum(pstd_values[-5:]) / 5
            if pstd_start - pstd_end > 10:
                problems.append(f"POLICY STD DROPPING FAST: {pstd_start:.1f} -> {pstd_end:.1f} deg")

    # Value loss instability
    vl_values = [u['value_loss'] for u in updates[-50:]]
    if vl_values:
        vl_max = max(vl_values)
        vl_min = min(vl_values)
        if vl_max > 10 * max(0.001, vl_min):
            problems.append(f"VALUE LOSS UNSTABLE: range {vl_min:.4f} - {vl_max:.4f} (ratio={vl_max/max(0.001,vl_min):.0f}x)")

    if problems:
        for p in problems:
            print(f"  [!] {p}")
    else:
        print("  No problems detected.")

    # =========================================================================
    # Summary Table (from periodic summaries)
    # =========================================================================
    if summaries:
        print("\n" + "-" * 70)
        print("PERIODIC SUMMARIES")
        print("-" * 70)
        print(f"  {'Upd':>6} {'Hours':>5} {'Gems':>6} {'Gem/hr':>7} {'OOB':>5} {'AvgRwd':>8} {'Best':>8} {'Ent':>6} {'Std':>5}")
        for s in summaries:
            print(f"  {s['update']:>6} {s.get('hours',0):>5.1f}h "
                  f"{s.get('total_gems','?'):>6} {s.get('gems_hr',0):>7.0f} "
                  f"{s.get('total_oob','?'):>5} {s.get('avg_reward',0):>8.1f} "
                  f"{s.get('best_reward',0):>8.1f} {s.get('entropy',0):>6.3f} "
                  f"{s.get('policy_std',0):>5.1f}")

    # =========================================================================
    # Final State
    # =========================================================================
    print("\n" + "-" * 70)
    print("FINAL STATE")
    print("-" * 70)
    u = updates[-1]
    print(f"  Update:     {u['update']}")
    print(f"  AvgRwd:     {u['avg_reward']:.1f}")
    print(f"  Entropy:    {u['entropy']:.3f}")
    if 'kl' in u:
        print(f"  KL:         {u['kl']:.4f}{' (KL-STOP)' if u.get('kl_stopped') else ''}")
    if u['policy_std']:
        print(f"  PolicyStd:  {u['policy_std']:.1f} deg")
    if episodes:
        e = episodes[-1]
        print(f"  Last Ep:    {e['episode']} (rwd={e['reward']:.0f}, gems={e['gems']}, OOB={e['oob']})")
    if summaries:
        s = summaries[-1]
        print(f"  Gems/hr:    {s.get('gems_hr', 0):.0f}")
        print(f"  Best avg:   {s.get('best_reward', 0):.1f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze PlatinumQuest PPO training logs')
    parser.add_argument('logfile', nargs='?', help='Log file to analyze (default: most recent)')
    parser.add_argument('--last', type=int, default=None, help='Only show last N episodes')
    parser.add_argument('--compare', action='store_true', help='Compare all log files')
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), 'logs')

    if args.compare:
        # Compare all logs
        files = sorted(glob.glob(os.path.join(log_dir, 'training_*.log')))
        if not files:
            print("No log files found in logs/")
            return

        print("=" * 90)
        print("LOG COMPARISON")
        print("=" * 90)
        print(f"  {'Log File':<35} {'Updates':>8} {'Eps':>5} {'AvgRwd':>8} {'Best':>8} {'Gem/hr':>7} {'OOB':>5} {'Ent':>6}")
        print("-" * 90)

        for f in files:
            data = parse_log(f)
            updates = data['updates']
            episodes = data['episodes']
            summaries = data['summaries']
            if not updates:
                continue

            name = os.path.basename(f)
            n_upd = len(updates)
            n_eps = len(episodes)
            avg_rwd = updates[-1]['avg_reward'] if updates else 0
            best = max(s.get('best_reward', 0) for s in summaries) if summaries else 0
            gems_hr = summaries[-1].get('gems_hr', 0) if summaries else 0
            total_oob = summaries[-1].get('total_oob', 0) if summaries else 0
            ent = updates[-1]['entropy'] if updates else 0

            print(f"  {name:<35} {n_upd:>8} {n_eps:>5} {avg_rwd:>8.0f} {best:>8.0f} {gems_hr:>7.0f} {total_oob:>5} {ent:>6.3f}")

        print("=" * 90)
        return

    if args.logfile:
        filepath = args.logfile
    else:
        # Find most recent
        files = sorted(glob.glob(os.path.join(log_dir, 'training_*.log')))
        if not files:
            print("No log files found in logs/")
            return
        filepath = files[-1]

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    data = parse_log(filepath)
    print_analysis(data, last_n=args.last)


if __name__ == '__main__':
    main()
