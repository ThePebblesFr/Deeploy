# End-to-end scenario testing runbook

How to exercise the full DynMultiDeeploy scenario system — the context-driven
DynReconf FSM plus all 4 baseline methods (Static-Small, Static-Large,
Round-Robin-Large, Priority-Sequential) and the α-sweep efficiency plot that
compares them. Everything below runs **inside the container**, from
`/app/Deeploy/DynMultiDeeployTest`:

```bash
docker start -i deeploy_main
cd /app/Deeploy/DynMultiDeeployTest
```

The worked examples use the `TinyClassifier`/`AnomalyDetection` pair
(`Tests/TinyClassifier_x_AnomalyDetection`, `--models TinyClassifier
AnomalyDetection`) that the rest of this doc assumes is already generated. If
you're starting from a different model pair, see [README.md](README.md) section 1
first, then come back here.

For the internals of any one piece (why a step exists, what it fixed, what its
known limitations are), see the referenced modules' own docstrings/comments —
this doc is the *how*, not the *why*.

---

## 0. One-time setup: generate the DynReconf binary

Only needed once per model pair (or after changing `Configs/*.json` or the model
directories themselves):

`--scenario` is required here too — the generated main() always `#include`s
`scenario_data.h`, which only gets (re)written when `--scenario` is given, so the
build fails without it unless a stale one happens to already exist. Generate a
throwaway one first (must match THIS pair's input shapes — an existing scenario
directory generated for a *different* model pair, e.g. this repo's own
`Scenarios/demo`, will have the wrong shapes and silently break the build):

```bash
python3 scenarioUtils/create_scenario.py \
  --task1-inputs-npz /app/Deeploy/DeeployTest/Tests/AnomalyDetection_light/inputs.npz \
  --task2-inputs-npz /app/Deeploy/DeeployTest/Tests/TinyClassifier_light/inputs.npz \
  --delta-t 12 --output-dir Scenarios/bootstrap --seed 0

python3 test_dynamic_2_models.py \
  -t Tests/TinyClassifier_x_AnomalyDetection \
  --models TinyClassifier AnomalyDetection \
  --scenario Scenarios/bootstrap \
  --gen-step
```

This runs the per-config generation loop (4 modes: LOW_ENERGY, CRITICAL_TASK1,
CRITICAL_TASK2, BOTH_CRITICAL — from `Configs/tinyclassifier_anomalydetection_4mode.json`
by default, or pass `--configs <path.json>` for a different config list), fuses
them into one binary, and generates the mode-dispatch code from
`testUtils/ModeSwitchBodyGen.py`. Expect to see 4 warnings like:

```
[DynMultiDeeployTestRunner] Warning: config ...'s own sanity build/simulation failed, but its Network.c was produced; continuing
```

This is a known, already-deferred issue (bit-exact correctness varies by tiling
config for these custom-generated models) — harmless for everything measured in
this pipeline (hardware metrics are shape/schedule-driven, not value-driven), not
something to chase down here.

You do **not** need to rerun `--gen-step` again unless you change the model
directories or the config list — every step below reuses this same generated
binary.

---

## 1. Generate a scenario

```bash
python3 scenarioUtils/create_scenario.py \
  --task1-inputs-npz /app/Deeploy/DeeployTest/Tests/AnomalyDetection_light/inputs.npz \
  --task2-inputs-npz /app/Deeploy/DeeployTest/Tests/TinyClassifier_light/inputs.npz \
  --delta-t 100 \
  --output-dir Scenarios/my_scenario \
  --seed 42
```

Writes `Scenarios/my_scenario/{task1_inputs.npz, task2_inputs.npz,
scenario_config.json}`. `--task{1,2}-inputs-npz` are just reference samples used
for shape/dtype/value-range — any existing single-sample `inputs.npz` for that
task works, the light variant's is fine even when testing heavy-variant modes.

Useful variants:

- **A real, mixed-mode scenario** (mode decided live by the FSM from actual
  reconstruction-error/margin signals): omit `--force-mode` as above. Inspect
  `scenario_config.json` to see the generated operator-order timeline and
  thresholds before running anything.
- **A pinned single-mode scenario** (for Static-Small/Static-Large baselines, or
  for per-mode calibration data — see step 4 and step 6): add `--force-mode
  {LOW_ENERGY,CRITICAL_TASK1,CRITICAL_TASK2,BOTH_CRITICAL}`. This pins
  `operation_mode` for the *entire* run by exploiting the FSM's own priority
  order — genuinely "no mode switching for the whole run" (verify via `grep -c
  "Runtime InitNetwork" <log>` — should be exactly `1`), not just a label that
  happens not to change.

---

## 2. Run DynReconf and parse the results

```bash
python3 test_dynamic_2_models.py \
  -t Tests/TinyClassifier_x_AnomalyDetection \
  --models TinyClassifier AnomalyDetection \
  --scenario Scenarios/my_scenario \
  2>&1 | tee /tmp/run.log
```

Drop `--gen-step` here (it's only needed once, per step 0) — this fast path just
regenerates `scenario_data.h`, rebuilds, and resimulates; seconds instead of
minutes.

Parse into a per-tick CSV:

```bash
python3 scenarioUtils/parse_scenario_log.py \
  --log /tmp/run.log --output /tmp/report.csv \
  --test-dir Tests/TinyClassifier_x_AnomalyDetection
```

`--test-dir` (same value as `-t` above) enables θ (ops) lookup from the profiling
JSONs `--gen-step` already produced, adding `theta`/`cumulative_theta` columns and
a final `gamma_eff` (GOP/J) summary line. Omit it for just δ/φ/E (mode, cores,
cycles, power, energy, criticality context) with `theta` left blank. Add
`--configs <path.json>` if you generated with a non-default config list.

Known cosmetic artifact, already handled: a burst of buffered
`@power.measure_N@` lines can occasionally split one `Tick %d: operation mode...`
line's text across several physical lines. The parser strips these before
matching, so the CSV is complete; if you're reading the raw log by eye instead,
don't be alarmed by an occasional mangled decision line — the tick's `context:`
line right after it is unaffected and has the real data.

---

## 3. Static-Small / Static-Large baselines

No new generation needed — these reuse the exact same DynReconf binary from step
0, just with a pinned scenario (step 1's `--force-mode`):

```bash
# Static-Small = LOW_ENERGY pinned for the whole run
python3 scenarioUtils/create_scenario.py \
  --task1-inputs-npz /app/Deeploy/DeeployTest/Tests/AnomalyDetection_light/inputs.npz \
  --task2-inputs-npz /app/Deeploy/DeeployTest/Tests/TinyClassifier_light/inputs.npz \
  --delta-t 100 --output-dir Scenarios/static_small --seed 42 --force-mode LOW_ENERGY

# Static-Large = BOTH_CRITICAL pinned for the whole run
python3 scenarioUtils/create_scenario.py \
  --task1-inputs-npz /app/Deeploy/DeeployTest/Tests/AnomalyDetection_light/inputs.npz \
  --task2-inputs-npz /app/Deeploy/DeeployTest/Tests/TinyClassifier_light/inputs.npz \
  --delta-t 100 --output-dir Scenarios/static_large --seed 42 --force-mode BOTH_CRITICAL

for scenario in static_small static_large; do
  python3 test_dynamic_2_models.py \
    -t Tests/TinyClassifier_x_AnomalyDetection --models TinyClassifier AnomalyDetection \
    --scenario Scenarios/$scenario > /tmp/$scenario.log 2>&1
  python3 scenarioUtils/parse_scenario_log.py \
    --log /tmp/$scenario.log --output /tmp/$scenario.csv \
    --test-dir Tests/TinyClassifier_x_AnomalyDetection
done
```

Sanity check either one actually never switched modes: `grep -c "Runtime
InitNetwork" /tmp/static_small.log` should print `1`.

---

## 4. Round-Robin-Large

Genuinely different execution model — each heavy variant runs *alone* on the
full 8-core cluster, alternating every tick (never co-resident) — so it needs its
own one-time generation step:

```bash
python3 test_roundrobin.py \
  -t Tests/TinyClassifier_x_AnomalyDetection \
  --models TinyClassifier AnomalyDetection \
  --scenario Scenarios/static_large \
  --gen-step
```

(Any scenario works here — Round-Robin-Large ignores `operator_order`/thresholds
entirely, it just needs `DELTA_T`/input data from `scenario_data.h`. Reusing
`static_large` here is arbitrary, not meaningful the way it is for the FSM-driven
methods.)

`--gen-step` regenerates both full-cluster deployments (`{Task2Base}_heavy`,
`{Task1Base}_heavy`) from scratch — needed once, or after model changes; drop it
on later runs to just reuse the existing fused `Network.c`/`.h` with a new
scenario:

```bash
python3 test_roundrobin.py \
  -t Tests/TinyClassifier_x_AnomalyDetection \
  --models TinyClassifier AnomalyDetection \
  --scenario Scenarios/my_scenario > /tmp/rr.log 2>&1

python3 scenarioUtils/parse_roundrobin_log.py \
  --log /tmp/rr.log --output /tmp/rr_report.csv \
  --model1-tag TinyClassifier_heavy --model2-tag AnomalyDetection_heavy
```

(`-t` is still a required argument even without `--gen-step`, but it's unused in
that case — the generated sources reused are already fixed at
`TEST_SIRACUSA/Tests/RoundRobin_Large` regardless of what `-t` points at.)

---

## 5. Priority-Sequential

Reuses Round-Robin-Large's generated `Network.c`/`.h` as-is (run step 4's
`--gen-step` first if you haven't) — only the main-loop dispatch template
differs (true priority preemption: only the more-critical task runs each tick,
the other is skipped, not deferred):

```bash
python3 test_priority_sequential.py \
  --models TinyClassifier AnomalyDetection \
  --scenario Scenarios/my_scenario > /tmp/ps.log 2>&1

python3 scenarioUtils/parse_priority_sequential_log.py \
  --log /tmp/ps.log --output /tmp/ps_report.csv \
  --model1-tag TinyClassifier_heavy --model2-tag AnomalyDetection_heavy
```

Use a scenario with real (not forced) input data here — unlike Round-Robin-Large,
this method's task-selection logic genuinely depends on the reconstruction-error/
margin signals computed from real inference outputs each tick.

---

## 6. α-sweep efficiency plot (all 5 methods)

`scenarioUtils/alpha_sweep.py` needs one clean, **pure** (never-switches) 100-tick
calibration run per DynReconf mode — reuses Static-Small/Static-Large's runs for
2 of the 4, plus 2 more:

```bash
python3 scenarioUtils/create_scenario.py \
  --task1-inputs-npz /app/Deeploy/DeeployTest/Tests/AnomalyDetection_light/inputs.npz \
  --task2-inputs-npz /app/Deeploy/DeeployTest/Tests/TinyClassifier_light/inputs.npz \
  --delta-t 100 --output-dir Scenarios/force_critical_task1 --seed 42 --force-mode CRITICAL_TASK1
python3 scenarioUtils/create_scenario.py \
  --task1-inputs-npz /app/Deeploy/DeeployTest/Tests/AnomalyDetection_light/inputs.npz \
  --task2-inputs-npz /app/Deeploy/DeeployTest/Tests/TinyClassifier_light/inputs.npz \
  --delta-t 100 --output-dir Scenarios/force_critical_task2 --seed 42 --force-mode CRITICAL_TASK2

for scenario in force_critical_task1 force_critical_task2; do
  python3 test_dynamic_2_models.py \
    -t Tests/TinyClassifier_x_AnomalyDetection --models TinyClassifier AnomalyDetection \
    --scenario Scenarios/$scenario > /tmp/$scenario.log 2>&1
  python3 scenarioUtils/parse_scenario_log.py \
    --log /tmp/$scenario.log --output /tmp/$scenario.csv \
    --test-dir Tests/TinyClassifier_x_AnomalyDetection
done
```

Then, with `static_small.csv`/`static_large.csv` (step 3), `force_critical_task1
.csv`/`force_critical_task2.csv` (above), `rr_report.csv` (step 4), and
`ps_report.csv` (step 5) all on hand:

```bash
python3 scenarioUtils/alpha_sweep.py \
  --mode-low-energy /tmp/static_small.csv \
  --mode-critical-task1 /tmp/force_critical_task1.csv \
  --mode-critical-task2 /tmp/force_critical_task2.csv \
  --mode-both-critical /tmp/static_large.csv \
  --round-robin-large /tmp/rr_report.csv \
  --priority-sequential /tmp/ps_report.csv \
  --output /tmp/alpha_sweep.png
```

Prints where (if anywhere) DynReconf's γ_eff(α_LE) curve crosses each flat
baseline, and writes the plot. Copy it out to view: `docker cp
deeploy_main:/tmp/alpha_sweep.png .`

**Known finding, not a bug**: with the current `TinyClassifier`/`AnomalyDetection`
placeholder models, Round-Robin-Large and Priority-Sequential come out ~3.5x more
efficient (GOP/J) than DynReconf or either static baseline — almost certainly
because these are tiny synthetic MLPs, where co-residency's fixed per-model
overhead (tiling setup, DMA orchestration, splitting 8 cores into smaller
subsets) is proportionally much larger than it would be for the paper's real
target models. The subtler DynReconf-vs-Static-Large crossover the plot is
actually meant to show does exist (~α_LE = 0.86 with the current calibration) but
is a small effect next to that gap. Worth rechecking once real models are in
place.

---

## Troubleshooting

- **`RuntimeError: Found an error in ...` / `Errors: N out of M`** during
  generation (steps 0, 4): the known, already-deferred correctness-varies-by-
  tiling-config issue. Harmless here — confirm `Network.c` was still produced (the
  scripts already do this check and only abort if it's genuinely missing).
- **`clang-15` segfault / `Error 139`** during a build: observed once as a
  transient compiler crash (likely resource contention from parallel builds),
  unrelated to any generated code. Just rerun the same build command.
- **`Invalid access` at some 0x57575757-looking address**: this was the L2
  memory capacity ceiling (scenario inputs used to be static L2 arrays, capped
  scenarios to ~20-30 ticks) — fixed by streaming scenario inputs through
  HyperRAM instead (see `scenarioUtils/ScenarioHeaderGen.py`'s docstring). If you
  see this again, it's a regression, not the known/expected issue.
- **A `parse_*_log.py` script raises about a missing/non-pure mode, or a missing
  `ops` field**: for θ/`alpha_sweep.py`, profiling JSONs must exist and be fresh
  (`--gen-step` writes them unconditionally) — rerun `--gen-step` if you changed
  models after they were last generated. For `alpha_sweep.py` specifically, each
  `--mode-*` input must come from a genuinely pure single-mode run
  (`--force-mode`), not a mixed real scenario.
