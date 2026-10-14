# Next development-only contact-context motion experiment

Declared during the component evaluation; implementation and14contracts nowPASS. Largest-map batch32finiteGPUforward/backward uses1.318GBallocated,0updates. Sealed preparation/protocol and source ZIP preserved; bounded development fit started2026-10-13, max5000updates/300s, seed121014. No native evaluation or test-map reopening.
Implement only after that batch's runtime source freeze ends. No concurrent
GPU fitting with wall-budgeted native evaluation. This is an isolated physical
motion regression experiment, never PPO, a checkpoint promotion or a gameplay
safety claim.

## Why change the representation

The relative-geometry model improved the balanced development loss but still
predicted large jump impulses while airborne and made ordinary motion worse
than constant velocity. An AABB distance is only a lower bound on triangle
distance: nine of Concentric's21jump frames look near an AABB without being near
the actual triangle. Cube has no such mismatch but still fails, so exact
proximity alone is not an established remedy. Also test multiple independent
contact contexts and ordinary-motion-preserving checkpoint selection.

## Inputs and architecture

Preserve the immutable expanded dataset:111,083native16ms transitions,518trials,
all47,932operative/source triangles, physical materials, native planes and real
polygon-boundary flags. Six training maps and two development maps remain as
declared previously; Bowl/Vortex neural test evaluation stays closed. No map ID,
reward, future pose, within-step contact flags or named maneuver labels as input.

Retain the84state and38surface features, relative/radius-scale encoding and
known position drift baseline. For every face and measured BEFORE center,
append operative-triangle closest-point vector/distance in radius units and
nearest real source-boundary segment vector/distance/availability: nine extra
features. Use continuous atan scaling, no surface/range/top-k cutoff. Faces with
no real boundary receive an explicit availability0; do not pretend they have
a zero-distance physical edge. Degenerate triangles must support segment and
corner distance. Normals, materials and original vertices remain present.

Replace the single averaged contact context by four learned attention heads,
32dimensions per head, joined with the128dimensional physical state. Continuous
locality prior uses exact triangle distance, finite clipped logits and the
actualdt. Every face participates. Output is a nine-dimensional correction to
position drift, initially zero. This changes features, capacity and attention
alongside sampling/selection: disclose those confounds, not a clean isolated
architecture win. Preserve the original inference-cache refusal for models
with alternate forward semantics.

## Bounded fit and selection

Seed121014; AdamW learning rate0.0003, decay0.0001, normalized delta MSE;
batch32, maximum5,000updates,300seconds training-loop wall limit. Scheduled
development passes may extend total wall time; record actual duration and limit
scope. Uniform training map,90%ordinary/5%jump/5%held-use, fallback only when a
map lacks a stratum. This balances archived regression examples, never forces a
gameplay action. Evaluate update0/every200updates/final on development only.

An update is eligible only if ordinary-motion mean position, velocity and spin
errors on **each** development map are at most1.05times constant velocity plus
absolute1e-6in the reported component's units. This5%tolerance is a declared
engineering selection threshold, not a safety guarantee. Among eligible
updates, choose lowest equal-map/stratum normalized development MSE. Update0's
known baseline is the explicit reference. If no improved eligible update
exists, report failure or baseline-only; do not cherry-pick an ineligible final
update. Preserve every scheduled result and per-component tail errors.

Report ordinary/grounded-jump/airborne-jump/held-use errors, same-BEFORE exact
geometry proximity buckets, per-power counts, p95/max errors, finite-gradient
and largest-map memory/timing results. Post-hoc native contact labels may audit
errors but never become inputs. Counts are correlated fixture samples, not
independent evidence of mastered mechanics. No test-set reopening or
autoregressive claim. No integration merely because balanced loss improves.

## Before fitting

Use maintained fitter/test files, no disposable Python scripts. Check analytic
face/edge/corner cases, duplicate/zero-area fans, real versus artificial fan
boundaries, alignment with F64CollisionMap.closest_points, origin translation,
all-face inclusion, input purity, zero-readout baseline equality and cache
refusal. Largest-map finite forward/backward must fit the existing8GB GPU.
Seal implementation/protocol/source hashes before updates begin. Do not modify
the reviewed collision archives, expanded data, protected A2 or executables.

A useful one-step result is only a prerequisite. Longer physical rollouts,
effect/actuation state evolution, calibrated contact uncertainty and real gem
collection still require separate checks. The goal remains general rolling,
spin, rebound, walls, jumps and power modifiers, not a catalog of tricks.
