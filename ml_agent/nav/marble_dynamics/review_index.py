"""Build a local review index without combining different experiments' scores."""
import html
import json
from pathlib import Path

from nav.collision.mesh import atomic_json
from nav.collision.viewer import write_atomic
from nav.marble_dynamics.record import ML,sha


def main():
    viewer=ML/'collision_maps/viewer';output=viewer/'planning_progress_20261012'
    output.mkdir(exist_ok=True)
    stages=[
        ('Walking geometry, three maps','reliability_geometry_validation_20261012','geometry_validation_20261012'),
        ('Walking geometry, other seven maps','reliability_geometry_pool_20261012','geometry_pool_20261012'),
        ('Resource preservation','reliability_blast_preserve_validation_20261012','blast_preserve_validation_20261012'),
        ('Recharge, development','reliability_recharge_regression_20261012','recharge_goals_20261012'),
        ('Recharge guards','reliability_recharge_validation_controls_20261012','recharge_controls_20261012'),
        ('Landing, new captured start','reliability_landing_regression_20261012','landing_regression_20261012'),
        ('Landing, archived start','reliability_landing_regression_r2_20261012','landing_regression_r2_20261012'),
        ('Landing, fresh three-map validation','reliability_landing_validation_20261012','landing_validation_20261012'),
        ('All observed goals, Treasure Box development','reliability_global_treasure_20261012','global_treasure_20261012'),
        ('Optimized physics, fresh guard validation','reliability_optimized_guards_20261012','optimized_guards_20261012'),
        ('Short continuations, Concentric development','reliability_short_concentric_r2_20261012','short_concentric_20261012'),
        ('Short continuations, hard maps development','reliability_short_hard_maps_20261012','short_hard_maps_20261012'),
        ('Probe timeout fixed, Fractured development','reliability_short_fractured_r2_20261012','short_fractured_r2_20261012'),
        ('Every-candidate stopping, Cube development','reliability_stopping_cube_20261012','stopping_cube_20261012'),
        ('Every-candidate stopping, Concentric development','reliability_stopping_concentric_20261012','stopping_concentric_20261012'),
        ('Pickup stopping, Cube development','reliability_pickup_stopping_cube_20261012','pickup_stopping_cube_20261012'),
        ('Pickup stopping, Concentric development','reliability_pickup_stopping_concentric_20261012','pickup_stopping_concentric_20261012'),
        ('Preserved stopping schedule, Cube development','reliability_pickup_stopping_cube_r2_20261012','pickup_stopping_cube_r2_20261012'),
        ('Preserved stopping schedule, Concentric development','reliability_pickup_stopping_concentric_r2_20261012','pickup_stopping_concentric_r2_20261012'),
        ('16ms first batch: Concentric complete, Cube invalid','reliability_small_step_validation_20261012','small_step_validation_20261012'),
        ('16ms Cube retry, archived start','reliability_small_step_cube_r2_20261012','small_step_cube_r2_20261012'),
        ('16ms Fractured, fresh start','reliability_small_step_fractured_20261012','small_step_fractured_20261012'),
        ('16ms remaining-map coverage, group A','reliability_small_step_pool_a_20261012','small_step_pool_a_20261012'),
        ('16ms remaining-map coverage, group B','reliability_small_step_pool_b_20261012','small_step_pool_b_20261012'),
        ('16ms Cube, longer development run','reliability_small_step_cube_extended_20261012','small_step_cube_extended_20261012'),
        ('16ms Fractured, longer development run','reliability_small_step_fractured_extended_20261012','small_step_fractured_extended_20261012'),
        ('Checked landing destination fixed, Cube development','reliability_small_step_cube_endpoint_fixed_20261012','small_step_cube_endpoint_fixed_20261012'),
        ('16ms Cragmire, longer development run','reliability_small_step_cragmire_extended_20261012','small_step_cragmire_extended_20261012')]
    rows=[];sources=[]
    for label,directory,gallery in stages:
        root=ML/'dynamics_reports'/directory
        if not (root/'summary.json').exists():continue
        summary=json.loads((root/'summary.json').read_text());prepared=json.loads((root/'prepared.json').read_text())
        sources.append(dict(path=str(root/'summary.json'),sha256=sha(root/'summary.json')))
        link='../'+gallery+'/index.html' if (viewer/gallery/'index.html').exists() else '../../../dynamics_reports/'+directory+'/summary.json'
        for b in summary['branches']:
            status=b.get('status','valid');archived=bool(prepared.get('fixture_data'))
            if b.get('stop_reason')=='bridge_timing_contract_violation':status='invalid: bridge timeout'
            cells=['<a href="'+link+'">'+html.escape(label)+'</a>',b['map'],b['seed'],b['branch'],
                b.get('actual_points','invalid') if status=='valid' else status,b.get('falls','unknown'),b.get('decisions','unknown'),
                prepared['decisions'],b.get('stop_reason',b.get('reason',status)),
                'archived start' if archived else 'new captured start',
                str(prepared.get('physics_frame_ms',64))+'/'+str(prepared.get('decision_ms',64))+'ms',
                str(prepared.get('search_budget_s',0))+'s' if prepared.get('search_budget_s',0) else 'unbounded',
                round(b['elapsed_wall_s'],3) if b.get('elapsed_wall_s') is not None else 'unknown']
            rows.append('<tr>'+''.join('<td>'+str(c)+'</td>' if i==0 else '<td>'+html.escape(str(c))+'</td>' for i,c in enumerate(cells))+'</tr>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Map-aware mechanics and planning review</title>
<style>body{font:16px system-ui;background:#101923;color:#dce5ee;margin:32px;line-height:1.6}a{color:#79c8ff}p{max-width:1100px}td,th{padding:8px;border-bottom:1px solid #314353;text-align:left}table{border-collapse:collapse;font-size:14px}</style>
<h1>Map-aware mechanics and gem planning</h1>
<p>The analytical predictor uses the complete collision mesh plus observed position, velocity, spin, native parameters, pending inputs and inventory. It checks proposed ordinary controls before execution. This includes rolling, jumping and impacts as well as verified powerup/Blast actuation. The frozen A2 checkpoint is unchanged. Separate bounded neural physics fits are recorded below. The contact-context fit preserves ordinary-motion means but badly mispredicts airborne jumps. Two subsequent varied-data/paired-action fits fail ordinary preservation and select only their unchanged update-zero baseline. None is used in gameplay. Six native power responses now expose script inventory, timed effects and Mega body growth; these are observation checks, not learned predictions. A separate fast full-map navigator has now been trained and evaluated. It remains experimental; no default checkpoint promotion has occurred.</p>
<ul><li><a href="../fast_navigator/index.html">Fast learned full-map navigator: complete ten-map native paths, scores and reference; candidate not promoted</a></li>
<li><a href="../../../docs/HANDOFF_FAST_NAVIGATOR_2026-10-13.md">Fast navigator implementation, training, bugs, failures and continuation runbook</a></li>
<li><a href="../index.html">All ten original collision maps: orbit, pan, zoom, inspect surfaces/materials</a></li>
<li><a href="../landing_continuation_20261012/index.html">Verified five-point flight plus supported landing: one complete forecast versus native motion</a></li>
<li><a href="../small_step_mechanics_20261012/index.html">16ms native mechanics: rolling, jump, powers1-5 and both verified Blast types</a></li>
<li><a href="../matched_mechanics_all_maps_20261012/index.html">Matched spin, rebound, jump timing and power-modifier experiments on all ten complete maps</a></li>
<li><a href="../rich_mechanics_20261012/index.html">Expanded ordinary controls and same-inventory power-use contrasts, including large contact-sensitive failures</a></li>
<li><a href="../../../dynamics_reports/native_motion_student_preparation_v2_20261012/unfitted_forward_feasibility.json">Unfitted full-map motion benchmark: exact input checks and GPU forward feasibility</a></li>
<li><a href="../../../dynamics_reports/native_motion_student_expanded_20261012/unfitted_forward_feasibility.json">Expanded native-motion preparation:518 trials,111,083 playable transitions, unfitted GPU checks</a></li>
<li><a href="../../../docs/REVIEW_PHYSICS_STUDENT_2026-10-12.md">Learned-motion benchmark scope, data gaps and proposed evaluation gates</a></li>
<li><a href="../../../docs/ROADMAP_FULL_MAP_LEARNING_2026-10-12.md">Path to continuous gem collection, training and higher full-round scores</a></li>
<li><a href="../../../docs/FULL_ROUND_REFERENCE_2026-10-12.md">Actual full-round A2 reference on all ten maps and historical-seed comparability</a></li>
<li><a href="../../../dynamics_reports/full_round_a2_pool_reference_20261012/summary.json">Native complete-round scores: all ten maps, one predeclared seed each</a></li>
<li><a href="../../../dynamics_reports/full_round_a2_replicated_20261012/summary.json">Completed legacy A2 replication: thirty full rounds, three seeds, ten maps</a></li>
<li><a href="../continuous_full_round_20261012/index.html">Continuous hybrid versus A2: complete Sprawl and KOTM round recordings</a></li>
<li><a href="../continuous_full_pool_r4_20261013/index.html">Completed ten-map pool: audited native scores, checked goals and disconnected respawn paths</a></li>
<li><a href="../geometry_component_pool_20261013/index.html">Completed ten-map component comparison: combined planning, geometry routing and frozen A2; all30 native recordings</a></li>
<li><a href="../../../docs/RESULT_GEOMETRY_COMPONENT_POOL_2026-10-13.md">Component result and costs:71.2/34.8/27.0 mean points, separate development seed</a></li>
<li><a href="../../../docs/RESULT_CONTACT_CONTEXT_STUDENT_2026-10-13.md">All-surface contact-context student: ordinary preservation passes, airborne jump forecasts still fail; not deployed</a></li>
<li><a href="../../../dynamics_reports/checked_recovery_native_preflight_20261013/summary.json">Checked recovery engineering preflight: Cube61/0falls, Duplex85/0falls; separate two-map result</a></li>
<li><a href="../checked_recovery_preflight_r2_20261013/index.html">Revised recovery engineering preflight: Cube28/0falls, Duplex70/4falls; corrected real-task rescue progress</a></li>
<li><a href="../../../dynamics_reports/checked_recovery_native_preflight_r2_20261013/chosen_reply_prefix_audit.json">215 native128ms executed-action prefix comparisons, including selected backups and power requests</a></li>
<li><a href="../../../docs/PLAN_GENERAL_MECHANICS_INTERVENTIONS_2026-10-13.md">Next general mechanics data plan: varied contact/spin/velocity with verified native interventions</a></li>
<li><a href="../../../docs/PLAN_CHECKED_RECOVERY_2026-10-13.md">Checked recovery protocol and real-task progress correction</a></li>
<li><a href="../checked_recovery_attempt_20261013/index.html">Recovery attempt: three complete rounds and the explicitly incomplete Bowl recording; physical versus real-task markers</a></li>
<li><a href="../../../docs/RESULT_CHECKED_RECOVERY_ATTEMPT_2026-10-13.md">Incomplete recovery comparison: Cube27/91points, Bowl900s wall-limit failure; no full-pool result</a></li>
<li><a href="../checked_recovery_task_cooldown_preflight_20261013/index.html">Task/cooldown recovery preflight: Cube47, Bowl9; substantial stopping time without pickups</a></li>
<li><a href="../../../docs/RESULT_RECOVERY_TASK_PREFLIGHT_2026-10-13.md">Verified recovery preflight and unsuccessful archived actor8 continuation; variant not integrated</a></li>
<li><a href="../../../dynamics_reports/checked_recovery_task_cooldown_preflight_20261013/chosen_reply_prefix_audit.json">675 native chosen-action128ms prefix comparisons</a></li>
<li><a href="../../../dynamics_reports/ordinary_recovery_actor8_archived_probe_20261013/result.json">All11 archived actor8 contexts, including seven deadline losses</a></li>
<li><a href="../../../docs/RESULT_NATIVE_FIXTURE_ROUNDTRIP_2026-10-13.md">Native physical snapshot, exact repeated prefixes and matched jump/brake/Super Speed responses</a></li>
<li><a href="../../../dynamics_reports/native_fixture_responses_20261013/result.json">All four physical-response pairs, including airborne jump with no impulse</a></li>
<li><a href="../../../dynamics_reports/native_fixture_transfer_pool_20261013/result.json">Native physical response transfer:24verifiedpairs across six training maps</a></li>
<li><a href="../../../datasets/marble_dynamics/native_inprocess_interventions_20261013/independent_audit.json">Varied ordinary-mechanics collection:72requests,66verifiedpairs,2153playable native transitions</a></li>
<li><a href="../../../docs/RESULT_PAIRED_MECHANICS_2026-10-13.md">Direct paired-action mechanics supervision and failed data-only fit; no policy promotion</a></li>
<li><a href="../../../docs/RESULT_SCRIPT_STATE_OBSERVABILITY_2026-10-13.md">Six power/script-state observation checks: Mega inventory alias and between-frame body transition verified</a></li>
<li><a href="../native_script_responses_20261013/index.html">Inspect all six native power responses, full Cube geometry and measured marble radius</a></li>
<li><a href="../../../dynamics_reports/observed_world_transition_prototype_20261013/manifest.json">378 observed-state transitions: current state and new reply inputs, next state targets; engineering prototype</a></li>
<li><a href="../../../dynamics_reports/native_scene_inventory_20261013/independent_audit.json">Ten-map passive native scene inventory: all snapshots preserve physical fields and clocks</a></li>
<li><a href="../../../dynamics_reports/native_state_coverage_preflight_20261013/independent_audit.json">Six held/released-use checks: live and native latch phases, initialized contacts and gravity/update context</a></li>
<li><a href="../../../dynamics_reports/observable_context_tokens_20261013/manifest.json">Semantic observable-context features:378verifiedexamples, fullCubegeometry, no raw object IDs</a></li>
<li><a href="../../../dynamics_reports/native_paired_open_loop_r3_20261013/independent_audit.json">Corrected physical baseline:90pairs/180branches/2496known-control forecast steps</a></li>
<li><a href="../../../docs/PROPOSAL_PHYSICS_BASELINE_ARCHITECTURE_2026-10-13.md">Review proposal: physics baseline, learned corrections and full-map gem decisions</a></li>
<li><a href="../../../docs/RESULT_CONTINUOUS_POOL_2026-10-13.md">Ten-map result and per-map falls, recovery and search cost</a></li>
<li><a href="../../../docs/PROTOCOL_GEOMETRY_COMPONENT_EVALUATION_2026-10-13.md">Declared geometry routing versus combined search component comparison</a></li>
<li><a href="../../../docs/RESULT_PHYSICS_STUDENT_2026-10-12.md">Bounded physics-fit results and limitations</a></li>
<li><a href="../../../dynamics_reports/native_motion_fit_residual_20261012/result.json">Development-only residual physics fit: learned corrections to velocity drift</a></li>
<li><a href="../../../dynamics_reports/full_round_reference_preparation_r4_20261012/protocol.json">Prepared three-seed full-round reference protocol: not yet executed</a></li>
<li><a href="../../../dynamics_reports/native_motion_fit_review_r2_20261012/protocol.json">Prepared bounded physical-motion fit: not yet executed</a></li>
<li><a href="../../../dynamics_reports/native_motion_complete_map_cache_20261012.json">All-surface inference cache: exact unfitted output equality, separate from planner speed</a></li>
<li><a href="../endpoint_braking_20261012/index.html">Inspect native endpoint failures: whole braking forecast and actual motion</a></li>
<li><a href="../../../dynamics_reports/small_step_terminal_brake_20261012/report.json">Exact Concentric endpoint replay: braking causes a native fall after1.088s</a></li>
<li><a href="../../../dynamics_reports/geometry_recovery_concentric_20261012/report.json">Concentric recovery diagnostic:972 proposals, no checked escape</a></li>
<li><a href="../recharge_goals_20261012/index.html">Live Concentric recharge loop: actual goals, charge and score</a></li>
<li><a href="../../../dynamics_reports/compiled_integration_20261012/comparison.json">Faster analytical sweeps: exact matched native controls, forecasts, pickups and landing</a></li>
<li><a href="../../../docs/HANDOFF_PLANNING_RELIABILITY_2026-10-12.md">Incremental engineering handoff and limitations</a></li></ul>
<p>Select a recording and use Focus trajectory and Time. Cyan is measured motion; coral is predicted motion. Episode views join short forecasts reanchored at observations; the isolated landing view shows one full open-loop forecast. Purple marks the actual goal or geometric landing destination. Scores are native game points, and charge comes from actual observations. Unknown final charge is left blank.</p>
<p>The separate legacy64ms A2 reference has single-seed mean17.2 and replicated mean24.93, distinct from historical32-33-point averages. The completed experimental16ms ten-map r4 pool averaged77.2 assisted versus30.4 frozen A2, with gains on allten maps in this one-seed comparison. All20native score/input audits pass, with zero implicit OOB keys. This composite teacher includes different routing, global goal/resource search and analytical forecasts; it is not newly trained policy competence or evidence of unseen-map generalization. Assisted rounds cost54.82wall minutes versus9.80for A2, with66/259falls and20/36acceptedrescues. Earlier two-map experiments and failures remain separate. KOTM is one equal-weight map and a modest dip there is no veto. The separate component comparison completed all30 rounds: combined planning71.2, geometry routing34.8, frozen A227.0 equal-map mean points, with128/204/313falls. This isolates a geometry-routing branch; the remaining combined-controller gain includes global goal/resource handling as well as forecasts. Checked recovery is being investigated as a separate follow-up.</p>
<p>Each short comparison in the table below restores one shared start before its two branches. No restore, teleport, rescue or item grant occurs during that scored play. A seed alone does not recreate an identical contact-sensitive spawn; archived-start runs are development regressions. These stages differ in settings and budgets: compare within a paired stage, not by pooling the rows. Falls are observed during the played interval; zero falls does not certify a complete round or every landing.</p>
<p>Physics/decision cadence and whole-search wall-time limits are explicit below. Wall seconds include search; simulated play remains64ms per decision. An invalid interrupted attempt has no comparable point score. The original Cube partial14/294 is retained as invalid, independently of the completed retry.</p>
<table><thead><tr><th>Stage</th><th>Map</th><th>Seed</th><th>Controller</th><th>Points</th><th>Falls</th><th>Played decisions</th><th>Budget</th><th>Stop</th><th>Start provenance</th><th>Physics/decision</th><th>Search limit</th><th>Wall seconds</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<p>Remaining limitations: some contact-sensitive futures are refused; broad long-horizon prediction tails are not calibrated. Cold searches take seconds. Regular and item-charged Blast impulses are verified, but later item pickups/respawns require observed replanning. Triple Blast, Mega actuation, Fireball and cannon events are not yet verified. Engineering, bounded physics learning and native evaluations continue under operator authorization; long policy training and checkpoint promotion remain paused.</p></html>'''
    page=page.replace('</table>', '</table><p>The expanded mechanics fixtures expose much larger forecast failures after fragile contacts, including64u in one6-second flight and15u in another. The largest case is flagged at the first divergent contact. One unflagged trajectory grows to0.061u despite near-exact isolated steps. These are preserved diagnostic limits, not scored-play outcomes or calibrated uncertainty.</p>')
    write_atomic(output/'index.html',page);atomic_json(output/'sources.json',dict(experiments=sources,training=False))
    print(output/'index.html')


if __name__=='__main__':main()
