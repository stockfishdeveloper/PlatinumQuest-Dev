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
    games = []  # [GAME END] lines with gap penalty data

    # Regex patterns
    upd_re = re.compile(
        r'Upd\s+(\d+)\s*\|\s*'
        r'PL=([-\d.]+)\s+VL=([-\d.]+)\s+'
        r'Ent=([-\d.]+)\s+GN=([-\d.]+)\s+'
        r'(?:CGN=([-\d.]+)\s+)?'
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
    game_end_re = re.compile(
        r'\[GAME END\] total gems this game: (\d+)pts \(best: (\d+)\) avg_gap_penalty: ([\d.]+)'
        r'(?:\s+near_misses: (\d+)\s+dwell_steps: (\d+))?'
        r'(?:\s+jump_rate: ([\d.]+)%(?:\s+brake_rate: ([\d.]+)%)?\s+avg_steps/gem: (\d+))?'
        r'(?:\s+overshoot: ([\d.]+)\s+\(([\d.]+)% of steps,\s+avg ([\d.]+)/frame\))?'
        r'(?:\s+pickup_speed: ([\d.]+)\s+brakes_near_pickup: ([\d.]+)/gem\s+slow_pickup_bonus: ([\d.]+))?'
    )
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
                tail = m.group(10)
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
                    'critic_grad_norm': float(m.group(6)) if m.group(6) else None,
                    'kl': float(m.group(7)),
                    'avg_reward': float(m.group(8)),
                    'avg_len': int(m.group(9)),
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

            # Game end lines (gap penalty + overshoot diagnostics)
            m = game_end_re.search(line)
            if m:
                games.append({
                    'gems': int(m.group(1)),
                    'best': int(m.group(2)),
                    'avg_gap_penalty': float(m.group(3)),
                    'near_misses': int(m.group(4)) if m.group(4) else None,
                    'dwell_steps': int(m.group(5)) if m.group(5) else None,
                    'jump_rate': float(m.group(6)) if m.group(6) else None,
                    'brake_rate': float(m.group(7)) if m.group(7) else None,
                    'avg_steps_per_gem': int(m.group(8)) if m.group(8) else None,
                    'overshoot_total': float(m.group(9)) if m.group(9) else None,
                    'overshoot_pct_steps': float(m.group(10)) if m.group(10) else None,
                    'overshoot_avg_per_frame': float(m.group(11)) if m.group(11) else None,
                    'pickup_speed': float(m.group(12)) if m.group(12) else None,
                    'brakes_near_pickup': float(m.group(13)) if m.group(13) else None,
                    'slow_pickup_bonus': float(m.group(14)) if m.group(14) else None,
                })

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
        'games': games,
        'load_info': load_info,
        'filepath': filepath,
    }


def print_analysis(data, last_n=None):
    """Print a comprehensive analysis of parsed log data."""
    updates = data['updates']
    episodes = data['episodes']
    summaries = data['summaries']
    games = data['games']
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

    has_cgn = any(u.get('critic_grad_norm') is not None for u in updates)
    cgn_hdr = f" {'CGN':>7}" if has_cgn else ""
    print(f"  {'Upd':>6} {'Ent':>7} {'PStd':>6} {'PL':>8} {'VL':>8} {'GN':>7}{cgn_hdr} {'KL':>7} {'AvgRwd':>9} {'OOB':>4} {'Gems':>5}")
    for i in sample_indices:
        u = updates[i]
        pstd = f"{u['policy_std']:.1f}" if u['policy_std'] else "?"
        kl_str = f"{u['kl']:.4f}" if 'kl' in u else "?"
        kl_flag = "*" if u.get('kl_stopped') else ""
        cgn_str = f" {u['critic_grad_norm']:>7.3f}" if has_cgn and u.get('critic_grad_norm') is not None else (" " * 8 if has_cgn else "")
        print(f"  {u['update']:>6} {u['entropy']:>7.3f} {pstd:>6} "
              f"{u['policy_loss']:>8.4f} {u['value_loss']:>8.4f} {u['grad_norm']:>7.3f}"
              f"{cgn_str} "
              f"{kl_str:>7}{kl_flag} "
              f"{u['avg_reward']:>9.1f} {u['oob']:>4} {u['gems']:>5}")

    # =========================================================================
    # Entropy / PolicyStd Trend
    # =========================================================================
    ent_values = [u['entropy'] for u in updates]
    pstd_values = [u['policy_std'] for u in updates if u['policy_std'] is not None]

    if len(ent_values) >= 20:
        print("\n" + "-" * 70)
        print("ENTROPY / POLICY STD TREND")
        print("-" * 70)

        # Split into fifths for trend analysis
        n = len(ent_values)
        fifth = n // 5
        if fifth >= 2:
            ent_fifths = []
            for i in range(5):
                start = i * fifth
                end = start + fifth if i < 4 else n
                ent_fifths.append(sum(ent_values[start:end]) / (end - start))
            print(f"  Entropy by fifth:  {' -> '.join(f'{v:.3f}' for v in ent_fifths)}")

            # Direction and rate
            ent_start_avg = sum(ent_values[:10]) / 10
            ent_end_avg = sum(ent_values[-10:]) / 10
            ent_delta = ent_end_avg - ent_start_avg
            ent_per_100 = ent_delta / (n / 100) if n > 0 else 0
            direction = "RISING" if ent_delta > 0.005 else "FALLING" if ent_delta < -0.005 else "STABLE"
            print(f"  Entropy trend:     {ent_start_avg:.3f} -> {ent_end_avg:.3f} ({direction}, {ent_per_100:+.4f}/100 updates)")

        if len(pstd_values) >= 20:
            pstd_n = len(pstd_values)
            pstd_fifth = pstd_n // 5
            if pstd_fifth >= 2:
                pstd_fifths = []
                for i in range(5):
                    start = i * pstd_fifth
                    end = start + pstd_fifth if i < 4 else pstd_n
                    pstd_fifths.append(sum(pstd_values[start:end]) / (end - start))
                print(f"  PolicyStd by fifth: {' -> '.join(f'{v:.1f}' for v in pstd_fifths)} deg")

            pstd_start = sum(pstd_values[:10]) / 10
            pstd_end = sum(pstd_values[-10:]) / 10
            pstd_delta = pstd_end - pstd_start
            pstd_per_100 = pstd_delta / (pstd_n / 100) if pstd_n > 0 else 0
            direction = "WIDENING" if pstd_delta > 0.5 else "NARROWING" if pstd_delta < -0.5 else "STABLE"
            print(f"  PolicyStd trend:   {pstd_start:.1f} -> {pstd_end:.1f} deg ({direction}, {pstd_per_100:+.2f} deg/100 updates)")

    # =========================================================================
    # Gap Penalty Analysis
    # =========================================================================
    if games:
        print("\n" + "-" * 70)
        print("GAP PENALTY ANALYSIS")
        print("-" * 70)

        gap_penalties = [g['avg_gap_penalty'] for g in games]
        game_gems = [g['gems'] for g in games]

        print(f"  Games tracked: {len(games)}")
        print(f"  Avg gap penalty:  min={min(gap_penalties):.1f}  max={max(gap_penalties):.1f}  "
              f"avg={sum(gap_penalties)/len(gap_penalties):.1f}  median={sorted(gap_penalties)[len(gap_penalties)//2]:.1f}")

        # High gap penalty games (likely overshoots/inefficiency)
        high_gap = [g for g in games if g['avg_gap_penalty'] > 50]
        if high_gap:
            pct = len(high_gap) / len(games) * 100
            avg_gems_high = sum(g['gems'] for g in high_gap) / len(high_gap)
            avg_gems_normal = sum(g['gems'] for g in games if g['avg_gap_penalty'] <= 50) / max(1, len(games) - len(high_gap))
            print(f"  High penalty games (>50): {len(high_gap)} ({pct:.0f}%) -- avg gems: {avg_gems_high:.0f} vs normal: {avg_gems_normal:.0f}")

        # Trend: first half vs second half
        if len(games) >= 6:
            half = len(games) // 2
            gap_first = sum(g['avg_gap_penalty'] for g in games[:half]) / half
            gap_second = sum(g['avg_gap_penalty'] for g in games[half:]) / (len(games) - half)
            direction = "UP" if gap_second > gap_first * 1.05 else "DOWN" if gap_second < gap_first * 0.95 else "flat"
            print(f"  Gap penalty trend: {gap_first:.1f} -> {gap_second:.1f} ({direction})")

        # Near-miss analysis (only if data available)
        near_miss_games = [g for g in games if g.get('near_misses') is not None]
        if near_miss_games:
            nm_vals = [g['near_misses'] for g in near_miss_games]
            dw_vals = [g['dwell_steps'] for g in near_miss_games]
            print(f"\n  Near misses/game: min={min(nm_vals)}  max={max(nm_vals)}  "
                  f"avg={sum(nm_vals)/len(nm_vals):.1f}  median={sorted(nm_vals)[len(nm_vals)//2]}")
            print(f"  Dwell steps/game: min={min(dw_vals)}  max={max(dw_vals)}  "
                  f"avg={sum(dw_vals)/len(dw_vals):.0f}  median={sorted(dw_vals)[len(dw_vals)//2]}")

            if len(near_miss_games) >= 6:
                half = len(near_miss_games) // 2
                nm_first = sum(g['near_misses'] for g in near_miss_games[:half]) / half
                nm_second = sum(g['near_misses'] for g in near_miss_games[half:]) / (len(near_miss_games) - half)
                dw_first = sum(g['dwell_steps'] for g in near_miss_games[:half]) / half
                dw_second = sum(g['dwell_steps'] for g in near_miss_games[half:]) / (len(near_miss_games) - half)
                nm_dir = "UP" if nm_second > nm_first * 1.05 else "DOWN" if nm_second < nm_first * 0.95 else "flat"
                dw_dir = "UP" if dw_second > dw_first * 1.05 else "DOWN" if dw_second < dw_first * 0.95 else "flat"
                print(f"  Near miss trend:  {nm_first:.1f} -> {nm_second:.1f} ({nm_dir})")
                print(f"  Dwell step trend: {dw_first:.0f} -> {dw_second:.0f} ({dw_dir})")

        # Overshoot penalty analysis (Time-Optimal Control diagnostics)
        overshoot_games = [g for g in games if g.get('overshoot_total') is not None]
        if overshoot_games:
            print(f"\n  OVERSHOOT PENALTY (Time-Optimal Control)")
            ov_total = [g['overshoot_total'] for g in overshoot_games]
            ov_pct = [g['overshoot_pct_steps'] for g in overshoot_games]
            ov_avg = [g['overshoot_avg_per_frame'] for g in overshoot_games]
            print(f"  Total/game:    min={min(ov_total):.0f}  max={max(ov_total):.0f}  avg={sum(ov_total)/len(ov_total):.0f}  median={sorted(ov_total)[len(ov_total)//2]:.0f}")
            print(f"  % of steps:    min={min(ov_pct):.1f}%  max={max(ov_pct):.1f}%  avg={sum(ov_pct)/len(ov_pct):.1f}%   (frames in overshoot zone)")
            print(f"  Severity:      avg={sum(ov_avg)/len(ov_avg):.2f} raw/frame    (low=grazing safe edge; high=deep overshoot)")
            if len(overshoot_games) >= 6:
                half = len(overshoot_games) // 2
                ov_first = sum(g['overshoot_total'] for g in overshoot_games[:half]) / half
                ov_second = sum(g['overshoot_total'] for g in overshoot_games[half:]) / (len(overshoot_games) - half)
                pct_first = sum(g['overshoot_pct_steps'] for g in overshoot_games[:half]) / half
                pct_second = sum(g['overshoot_pct_steps'] for g in overshoot_games[half:]) / (len(overshoot_games) - half)
                ov_dir = "UP" if ov_second > ov_first * 1.05 else "DOWN" if ov_second < ov_first * 0.95 else "flat"
                pct_dir = "UP" if pct_second > pct_first * 1.05 else "DOWN" if pct_second < pct_first * 0.95 else "flat"
                print(f"  Total trend:   {ov_first:.0f} -> {ov_second:.0f} ({ov_dir})    (DOWN = model learning to brake earlier)")
                print(f"  %steps trend:  {pct_first:.1f}% -> {pct_second:.1f}% ({pct_dir})")

        # Brake-effectiveness diagnostics
        brake_games = [g for g in games if g.get('pickup_speed') is not None]
        if brake_games:
            print(f"\n  BRAKE EFFECTIVENESS")
            ps = [g['pickup_speed'] for g in brake_games]
            bnp = [g['brakes_near_pickup'] for g in brake_games]
            spb = [g['slow_pickup_bonus'] for g in brake_games]
            print(f"  Pickup speed:        min={min(ps):.1f}  max={max(ps):.1f}  avg={sum(ps)/len(ps):.1f}    (OUTCOME: should DROP if brake works)")
            print(f"  Brakes/gem (last30): min={min(bnp):.2f}  max={max(bnp):.2f}  avg={sum(bnp)/len(bnp):.2f}    (MECHANISM: should RISE if model learned timing)")
            print(f"  Slow-pickup bonus:   min={min(spb):.0f}   max={max(spb):.0f}   avg={sum(spb)/len(spb):.0f}     (per-game total reward earned)")
            if len(brake_games) >= 6:
                half = len(brake_games) // 2
                ps_first = sum(g['pickup_speed'] for g in brake_games[:half]) / half
                ps_second = sum(g['pickup_speed'] for g in brake_games[half:]) / (len(brake_games) - half)
                bnp_first = sum(g['brakes_near_pickup'] for g in brake_games[:half]) / half
                bnp_second = sum(g['brakes_near_pickup'] for g in brake_games[half:]) / (len(brake_games) - half)
                ps_dir = "DOWN" if ps_second < ps_first * 0.95 else "UP" if ps_second > ps_first * 1.05 else "flat"
                bnp_dir = "UP" if bnp_second > bnp_first * 1.05 else "DOWN" if bnp_second < bnp_first * 0.95 else "flat"
                print(f"  Pickup speed trend:    {ps_first:.1f} -> {ps_second:.1f} ({ps_dir})    (DOWN = WORKING)")
                print(f"  Brakes/gem trend:      {bnp_first:.2f} -> {bnp_second:.2f} ({bnp_dir})  (UP = model learned brake timing)")

        # Correlation between gems and gap penalty
        if len(games) >= 5:
            n = len(games)
            mean_g = sum(game_gems) / n
            mean_p = sum(gap_penalties) / n
            cov = sum((game_gems[i] - mean_g) * (gap_penalties[i] - mean_p) for i in range(n)) / n
            std_g = (sum((g - mean_g)**2 for g in game_gems) / n) ** 0.5
            std_p = (sum((p - mean_p)**2 for p in gap_penalties) / n) ** 0.5
            corr = cov / (std_g * std_p) if std_g > 0 and std_p > 0 else 0
            print(f"  Gems vs gap penalty correlation: {corr:.2f} (negative = higher penalty when fewer gems)")

    # =========================================================================
    # Problem Detection
    # =========================================================================
    print("\n" + "-" * 70)
    print("PROBLEM DETECTION")
    print("-" * 70)

    problems = []

    # Entropy collapse
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
        elif ent_drop > 0.02:
            problems.append(f"ENTROPY DECLINING: {ent_start:.3f} -> {ent_end:.3f} (delta={ent_drop:.3f}) - monitor for death spiral")
        elif ent_drop < -0.02:
            problems.append(f"ENTROPY RISING: {ent_start:.3f} -> {ent_end:.3f} (delta={abs(ent_drop):.3f}) - may need lower entropy_coef")

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

    # Critic gradient norm spikes
    cgn_values = [u['critic_grad_norm'] for u in updates if u.get('critic_grad_norm') is not None]
    if cgn_values:
        cgn_spikes = [v for v in cgn_values if v > 100]
        if cgn_spikes:
            problems.append(f"CRITIC GRAD NORM SPIKES: {len(cgn_spikes)} updates with CGN>100 (max={max(cgn_spikes):.1f})")

    # Gap penalty trend
    if games and len(games) >= 6:
        half = len(games) // 2
        gap_first = sum(g['avg_gap_penalty'] for g in games[:half]) / half
        gap_second = sum(g['avg_gap_penalty'] for g in games[half:]) / (len(games) - half)
        if gap_second > gap_first * 1.3:
            problems.append(f"GAP PENALTY RISING: {gap_first:.1f} -> {gap_second:.1f} (agent getting less efficient)")

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
    print(f"  GradNorm:   {u['grad_norm']:.3f}")
    if u.get('critic_grad_norm') is not None:
        print(f"  CriticGN:   {u['critic_grad_norm']:.3f}")
    if episodes:
        e = episodes[-1]
        print(f"  Last Ep:    {e['episode']} (rwd={e['reward']:.0f}, gems={e['gems']}, OOB={e['oob']})")
    if summaries:
        s = summaries[-1]
        print(f"  Gems/hr:    {s.get('gems_hr', 0):.0f}")
        print(f"  Best avg:   {s.get('best_reward', 0):.1f}")
    if games:
        g = games[-1]
        recent_gap = sum(x['avg_gap_penalty'] for x in games[-10:]) / min(10, len(games))
        print(f"  Last game:  {g['gems']}pts, gap_penalty={g['avg_gap_penalty']:.1f}")
        print(f"  Avg gap (last 10 games): {recent_gap:.1f}")

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
