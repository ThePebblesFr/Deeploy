#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generates a scenario for the DynMultiDeeploy context-driven FSM: DELTA_T random
input samples per task, an operator's-order timeline, and a scenario_config.json
carrying thresholds and per-mode power estimates.

Usage:
    python3 create_scenario.py \
        --task1-inputs-npz /app/Deeploy/DeeployTest/Tests/AnomalyDetection_light/inputs.npz \
        --task2-inputs-npz Tests/miniMNV2_x_AnomalyDetection/miniMNV2/inputs.npz \
        --delta-t 12 \
        --output-dir Scenarios/demo \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

# Default per-mode power estimates (Watts), calibrated from a real GVSoC --power run
# of the miniMNV2/AnomalyDetection placeholder pair (10 sequential @power.measure_N@
# readings averaged per mode: LOW_ENERGY over 5 inferences, CRITICAL_TASK2 over 2,
# CRITICAL_TASK1 over 1, BOTH_CRITICAL over 2). Task 1 = AnomalyDetection, Task 2 =
# miniMNV2 (image-classification stand-in) -- CRITICAL_TASK1 is the config that gives
# AnomalyDetection more cores (physically "2_6": miniMNV2=2, AnomalyDetection=6),
# CRITICAL_TASK2 gives miniMNV2 more cores (physically "6_2"). These feed the
# on-device cumulative-energy proxy only; the paper's actually-reported phi/E metrics
# come from GVSoC's own per-inference log lines, parsed post-hoc, not this estimate.
DEFAULT_PHI_ESTIMATE = {
    "LOW_ENERGY": 0.0002746,
    "CRITICAL_TASK1": 0.000216,
    "CRITICAL_TASK2": 0.0002955,
    "BOTH_CRITICAL": 0.000276,
}

OPERATOR_ORDER_MODES = {0: "none", 1: "CRITICAL_TASK1", 2: "CRITICAL_TASK2", 3: "BOTH_CRITICAL"}

FORCE_MODE_TO_OPERATOR_CMD = {"CRITICAL_TASK1": 1, "CRITICAL_TASK2": 2, "BOTH_CRITICAL": 3}
# Deliberately absurdly high: forcing a non-LOW_ENERGY mode for the whole run needs
# the power-over-budget override to never fire (it outranks operator_order in the
# FSM's priority), and a real cumulative estimate over any realistic DELTA_T is
# orders of magnitude below this (observed: ~50 uJ over 100 ticks in LOW_ENERGY,
# the cheapest mode -- even a heavier mode over many more ticks stays the same
# order of magnitude).
FORCE_MODE_ENERGY_THRESHOLD_UJ = 1e9


def load_reference_sample(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    key = "input_0" if "input_0" in d.files else d.files[0]
    return d[key]


def generate_task_inputs(reference: np.ndarray, delta_t: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates DELTA_T random samples with the same shape/dtype/value-range as the
    reference single-sample input.npz, stacked along a new leading time axis.
    """
    sample_shape = reference.shape
    lo, hi = int(reference.min()), int(reference.max())
    if lo == hi:
        # Degenerate reference sample (e.g. all-zero) -- fall back to the dtype's
        # natural signed/unsigned int8 range so the generated inputs aren't all zero.
        lo, hi = (0, 255) if reference.dtype.kind == "u" or lo >= 0 else (-128, 127)

    samples = rng.integers(lo, hi + 1, size=(delta_t, *sample_shape), dtype=np.int64)
    return samples.astype(reference.dtype)


def generate_operator_order(delta_t: int, rng: random.Random, min_burst: int = 2, max_burst: int = 4) -> list:
    """
    Mostly-zero (no operator order) timeline, with exactly one contiguous non-zero
    burst injected at a random position, per the "should appear at least once...
    for a few inferences at least" requirement.
    """
    order = [0] * delta_t
    if delta_t == 0:
        return order

    burst_len = min(rng.randint(min_burst, max_burst), delta_t)
    burst_start = rng.randint(0, delta_t - burst_len)
    burst_value = rng.choice([1, 2, 3])
    for t in range(burst_start, burst_start + burst_len):
        order[t] = burst_value

    return order


def main():
    parser = argparse.ArgumentParser(description="Generate a DynMultiDeeploy scenario.")
    parser.add_argument("--task1-inputs-npz", type=str, required=True,
                        help="Path to a reference single-sample inputs.npz for task 1 (shape/dtype/range template).")
    parser.add_argument("--task2-inputs-npz", type=str, required=True,
                        help="Path to a reference single-sample inputs.npz for task 2.")
    parser.add_argument("--delta-t", type=int, required=True, help="Number of scenario ticks.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write the scenario into.")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--task1-margin-threshold-light", type=int, default=1,
                        help="Task-1 (AnomalyDetection light) critical when (top logit - runner-up "
                             "logit) falls below this. AnomalyDetection is a 2-class classifier "
                             "(critical vs normal), so this uses the same top-vs-runner-up margin "
                             "proxy as task 2, not a reconstruction-error signal -- there is no "
                             "reconstruction error to compute against a 2-element output.")
    parser.add_argument("--task1-margin-threshold-heavy", type=int, default=1)
    parser.add_argument("--task2-margin-threshold", type=int, default=1,
                        help="Task-2 critical when (top logit - runner-up logit) falls below this.")
    parser.add_argument("--energy-threshold-uj", type=float, default=100.0,
                        help="Cumulative on-device energy-estimate threshold (microjoules) forcing LOW_ENERGY.")
    parser.add_argument("--force-mode", type=str, default=None,
                        choices=["LOW_ENERGY", "CRITICAL_TASK1", "CRITICAL_TASK2", "BOTH_CRITICAL"],
                        help="Pin operation_mode to this value for the ENTIRE run instead of a real "
                             "context-driven timeline -- a genuine non-switching static baseline (Phase 5's "
                             "Static-Small = LOW_ENERGY, Static-Large = BOTH_CRITICAL), built by exploiting the "
                             "FSM's own priority order rather than a separate non-FSM binary: LOW_ENERGY forces "
                             "--energy-threshold-uj to 0 (power-over-budget, top priority, true from tick 1 "
                             "onward); the other 3 force a constant operator_order matching that mode (2nd "
                             "priority) and raise the energy threshold so high power-over-budget can never "
                             "preempt it. --energy-threshold-uj/the random operator-order burst are ignored "
                             "when this is given.")

    args = parser.parse_args()

    np_rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    task1_ref = load_reference_sample(Path(args.task1_inputs_npz))
    task2_ref = load_reference_sample(Path(args.task2_inputs_npz))

    task1_inputs = generate_task_inputs(task1_ref, args.delta_t, np_rng)
    task2_inputs = generate_task_inputs(task2_ref, args.delta_t, np_rng)

    if args.force_mode == "LOW_ENERGY":
        operator_order = [0] * args.delta_t
        energy_threshold_uj = 0.0
    elif args.force_mode:
        operator_order = [FORCE_MODE_TO_OPERATOR_CMD[args.force_mode]] * args.delta_t
        energy_threshold_uj = FORCE_MODE_ENERGY_THRESHOLD_UJ
    else:
        operator_order = generate_operator_order(args.delta_t, py_rng)
        energy_threshold_uj = args.energy_threshold_uj

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_dir / "task1_inputs.npz", inputs=task1_inputs)
    np.savez(output_dir / "task2_inputs.npz", inputs=task2_inputs)

    scenario_config = {
        "delta_t": args.delta_t,
        "task1_input_shape": list(task1_ref.shape),
        "task2_input_shape": list(task2_ref.shape),
        "thresholds": {
            "task1_margin_threshold_light": args.task1_margin_threshold_light,
            "task1_margin_threshold_heavy": args.task1_margin_threshold_heavy,
            "task2_margin_threshold": args.task2_margin_threshold,
            "energy_threshold_uj": energy_threshold_uj,
        },
        "phi_estimate_watts": DEFAULT_PHI_ESTIMATE,
        "operator_order": operator_order,
    }

    with open(output_dir / "scenario_config.json", "w") as f:
        json.dump(scenario_config, f, indent=2)

    print(f"[create_scenario] Generated scenario at {output_dir}")
    print(f"[create_scenario] delta_t={args.delta_t}, task1_shape={task1_ref.shape}, task2_shape={task2_ref.shape}")
    if args.force_mode:
        print(f"[create_scenario] force-mode={args.force_mode} for all {args.delta_t} ticks "
              f"(energy_threshold_uj={energy_threshold_uj})")
    else:
        burst_ticks = [t for t, v in enumerate(operator_order) if v != 0]
        if burst_ticks:
            mode_name = OPERATOR_ORDER_MODES[operator_order[burst_ticks[0]]]
            print(f"[create_scenario] operator order burst: ticks {burst_ticks[0]}-{burst_ticks[-1]} -> {mode_name}")


if __name__ == "__main__":
    main()
