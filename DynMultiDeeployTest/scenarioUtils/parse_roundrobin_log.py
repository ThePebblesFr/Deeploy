#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Parses a saved GVSoC Round-Robin-Large run log (roundrobin.c, built with
-D GVSOC_EXTRA_FLAGS=--power) into a per-tick CSV: each tick alternates one full
InitNetwork+RunNetwork call for Task 2 (model1_tag) and one for Task 1 (model2_tag),
both on the full 8-core cluster -- unlike the DynReconf/Static scenario logs, there
is no mode decision to parse, and InitNetwork runs on EVERY tick for BOTH models
(never skipped), so this script's job is much simpler: pair up 4 timed calls per
tick (Task2 Init, Task2 Run, Task1 Init, Task1 Run) and their power measurements.

Unlike every other template in this codebase, roundrobin.c brackets BOTH
InitNetwork and RunNetwork calls with the 0xabbaabba/0xdeadcaca power-measurement
markers (not just RunNetwork) -- InitNetwork's power/energy is NOT negligible here
the way it is for Static/DynReconf (where it runs once for the whole run): it reruns
every tick for both models, so omitting it would make Round-Robin look artificially
efficient. This means there are 4 @power.measure_N@ readings per tick here (Init,
Run, Init, Run in program order), not 1.

Usage:
    python3 test_roundrobin.py -t Tests/<pair> --models <m1> <m2> \
        --scenario Scenarios/static_large --gen-step 2>&1 | tee /tmp/rr.log
    python3 scenarioUtils/parse_roundrobin_log.py --log /tmp/rr.log --output /tmp/rr_report.csv \
        --test-dir Tests/RoundRobin_Large --model1-tag TinyClassifier_heavy --model2-tag AnomalyDetection_heavy
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MultiDeeploy.ModelProfile import ModelProfile

FREQUENCY_HZ = 50_000_000.0

TICK_RE = re.compile(r"\[HOST CORE\] Tick (\d+)")
INIT_CYCLES_RE = re.compile(r"\[(TASK1|TASK2)\] Runtime InitNetwork: (\d+) cycles")
RUN_CYCLES_RE = re.compile(r"\[(TASK1|TASK2)\] Runtime RunNetwork: (\d+) cycles")
POWER_RE = re.compile(r"@power\.measure_(\d+)@([\d.]+)@")


def _load_total_ops(profiling_json_path: Path) -> int:
    with open(profiling_json_path) as f:
        profile = ModelProfile.from_dict(json.load(f))
    total_ops = profile.total_ops()
    if total_ops is None:
        raise ValueError(f"{profiling_json_path} has no per-layer 'ops' field -- rerun --gen-step.")
    return total_ops


def parse_log(log_path: Path, theta_task2: Optional[int] = None, theta_task1: Optional[int] = None) -> List[Dict]:
    raw_text = log_path.read_text()

    phi_by_index: Dict[int, float] = {}
    for m in POWER_RE.finditer(raw_text):
        phi_by_index[int(m.group(1))] = float(m.group(2))

    # Same UART-buffering artifact fixed in parse_scenario_log.py: a burst of
    # buffered power lines can flush mid-printf, splitting one line's text across
    # several -- stripping each token (with its own trailing newline) restores the
    # original single-line text before per-line regex matching.
    cleaned_text = re.sub(r"@power\.measure_\d+@[\d.]+@\n?", "", raw_text)

    ticks: List[Dict] = []
    current: Optional[Dict] = None
    # Counts BOTH InitNetwork and RunNetwork calls in program order == power
    # measure index (roundrobin.c brackets both, unlike every other template here).
    call_index = 0

    for line in cleaned_text.splitlines():
        m = TICK_RE.search(line)
        if m:
            if current is not None:
                ticks.append(current)
            current = {
                "tick": int(m.group(1)),
                "task2_init_cycles": None, "task2_init_phi_watts": None,
                "task2_run_cycles": None, "task2_run_phi_watts": None,
                "task1_init_cycles": None, "task1_init_phi_watts": None,
                "task1_run_cycles": None, "task1_run_phi_watts": None,
            }
            continue

        if current is None:
            continue

        m = INIT_CYCLES_RE.search(line)
        if m:
            task, cycles = m.group(1), int(m.group(2))
            key_prefix = "task2" if task == "TASK2" else "task1"
            current[f"{key_prefix}_init_cycles"] = cycles
            current[f"{key_prefix}_init_phi_watts"] = phi_by_index.get(call_index)
            call_index += 1
            continue

        m = RUN_CYCLES_RE.search(line)
        if m:
            task, cycles = m.group(1), int(m.group(2))
            key_prefix = "task2" if task == "TASK2" else "task1"
            current[f"{key_prefix}_run_cycles"] = cycles
            current[f"{key_prefix}_run_phi_watts"] = phi_by_index.get(call_index)
            call_index += 1
            continue

    if current is not None:
        ticks.append(current)

    theta_total = None
    if theta_task2 is not None and theta_task1 is not None:
        theta_total = theta_task2 + theta_task1

    cumulative_theta = 0
    for row in ticks:
        energies = []
        for prefix in ("task2", "task1"):
            for phase in ("init", "run"):
                cycles = row[f"{prefix}_{phase}_cycles"]
                phi = row[f"{prefix}_{phase}_phi_watts"]
                energy = (cycles / FREQUENCY_HZ) * phi if (cycles is not None and phi is not None) else None
                row[f"{prefix}_{phase}_energy_j"] = energy
                if energy is not None:
                    energies.append(energy)
        row["total_energy_j"] = sum(energies) if energies else None
        if theta_total is not None:
            cumulative_theta += theta_total
            row["theta"] = theta_total
            row["cumulative_theta"] = cumulative_theta
        else:
            row["theta"] = None
            row["cumulative_theta"] = None

    return ticks


def write_csv(ticks: List[Dict], output_path: Path) -> None:
    fieldnames = [
        "tick",
        "task2_init_cycles", "task2_init_phi_watts", "task2_init_energy_j",
        "task2_run_cycles", "task2_run_phi_watts", "task2_run_energy_j",
        "task1_init_cycles", "task1_init_phi_watts", "task1_init_energy_j",
        "task1_run_cycles", "task1_run_phi_watts", "task1_run_energy_j",
        "total_energy_j", "theta", "cumulative_theta",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ticks:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Parse a Round-Robin-Large GVSoC log into a per-tick CSV report.")
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="Tests/RoundRobin_Large",
                         help="Directory holding <model1-tag>/deeployStates/<model1-tag>_profiling.json etc. "
                              "(TEST_SIRACUSA/ prefix added automatically, matching test_roundrobin.py).")
    parser.add_argument("--model1-tag", type=str, default=None, help="e.g. TinyClassifier_heavy (Task 2).")
    parser.add_argument("--model2-tag", type=str, default=None, help="e.g. AnomalyDetection_heavy (Task 1).")
    args = parser.parse_args()

    theta_task2 = theta_task1 = None
    if args.model1_tag and args.model2_tag:
        base = Path("TEST_SIRACUSA") / args.test_dir
        theta_task2 = _load_total_ops(base / args.model1_tag / "deeployStates" / f"{args.model1_tag}_profiling.json")
        theta_task1 = _load_total_ops(base / args.model2_tag / "deeployStates" / f"{args.model2_tag}_profiling.json")
        print(f"[parse_roundrobin_log] theta_task2({args.model1_tag})={theta_task2}, "
              f"theta_task1({args.model2_tag})={theta_task1}, per-tick theta={theta_task2 + theta_task1}")

    ticks = parse_log(Path(args.log), theta_task2, theta_task1)
    if not ticks:
        raise RuntimeError(f"No '[HOST CORE] Tick N' lines found in {args.log}.")

    write_csv(ticks, Path(args.output))

    total_energy_j = sum(t["total_energy_j"] for t in ticks if t["total_energy_j"] is not None)
    print(f"[parse_roundrobin_log] Wrote {args.output} ({len(ticks)} ticks)")
    print(f"[parse_roundrobin_log] Total measured energy: {total_energy_j:.6e} J")

    if theta_task2 is not None:
        total_theta = ticks[-1]["cumulative_theta"]
        if total_theta is not None and total_energy_j > 0:
            gamma_eff = total_theta / total_energy_j
            print(f"[parse_roundrobin_log] Total theta: {total_theta} ops -- "
                  f"gamma_eff = {gamma_eff:.6e} OP/J ({gamma_eff / 1e9:.3f} GOP/J)")


if __name__ == "__main__":
    main()
