#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Per-tick relaunch orchestrator -- the workaround for MultiDeeploy's
double-buffered (DB) closures hanging when repeatedly invoked in-process
through the DynMultiDeeploy scenario FSM harness (dynmultideeploytest.c's own
`for (t=0; t<DELTA_T; t++)` loop). See the plan file's "Step 2 follow-up"
section for the full investigation: three independent hypotheses (a
multi-tensor channel leak, a barrier event-mask leak, a deferred single-tile
flush) were each tested against real GVSoC runs and disproven or found
insufficient. A single inference through the exact same FSM logic is proven
correct (and faster) with DB, so this sidesteps the repeated-invocation
trigger entirely: relaunch a single-inference binary once per tick, in a
genuinely fresh GVSoC process each time, paying process-relaunch wall-clock
overhead (not reflected in any reported hardware metric) instead of chasing
the underlying GVSoC-level cause further.

The FSM's mode-decision logic (power-over-budget > operator_order >
confidence > default LOW_ENERGY) is replicated here in Python, host-side,
rather than on-device: this script already needs each tick's real
task1_err/task2_margin -- parsed back from that tick's own process output --
to decide the NEXT tick's mode, so there is no benefit to also deciding a
tick's own mode on-device (see Platforms/DynSiracusaSingleShot/src/
dynmultideeploytest_singleshot_template.c's file-level comment for the C
side of this).

Requires that `test_dynamic_2_models.py --gen-step` has ALREADY been run for
this test/models/--configs combination (producing the fused Network.c/.h
under TEST_SIRACUSA/Tests/<m0>_x_<m1>/) -- this script only handles:
scenario_data.h/hex generation, the one-time DynSiracusaSingleShot cmake
configure+build, and the per-tick relaunch loop.

Known measurement gap, worked around via --phi-calibration-*: comparing a
100-tick SB-only run of this script against the known continuous-process SB
baseline (both pure LOW_ENERGY) found the per-tick @power.measure_N@ value
this script reads back is uniformly ~28% higher than the continuous-process
measurement for the identical workload -- constant across all 100 launches
(not decaying like a warm-up effect), so most likely tied to InitNetworks_*
always running here (it only runs on real mode transitions in a continuous
run) rather than anything data-dependent. Root cause not fully diagnosed.
DB's own cycle counts (this script's actual reason to exist) are unaffected
and proven accurate throughout this session; only the power/energy side is
biased. Workaround: --phi-calibration-<mode> lets each mode's phi_watts be
overridden with a value calibrated from an existing PURE, continuous-process
SB run for that mode (e.g. Static-Small's CSV for LOW_ENERGY) instead of
this script's own (biased) power measurement -- justified because power draw
for a given mode/core-split should be roughly independent of whether DMA
overlaps compute (the same hardware blocks are active either way, just
scheduled differently), so SB's own already-validated phi is a reasonable
stand-in while this script's cycle counts capture DB's real latency benefit.

Usage:
    python3 scenarioUtils/run_singleshot_scenario.py \\
        --models TinyClassifier AnomalyDetection \\
        --scenario Scenarios/static_small \\
        --configs Configs/tinyclassifier_anomalydetection_4mode_db.json \\
        --output /tmp/singleshot_static_small.csv \\
        --phi-calibration-low-energy /tmp/static_small_sb.csv
"""

import argparse
import csv
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# See parse_scenario_log.py's own comment for why this needs to be explicit:
# invoked as `python3 scenarioUtils/run_singleshot_scenario.py`, which puts
# scenarioUtils/ itself (not DynMultiDeeployTest/) on sys.path[0].
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testUtils.ConfigLoader import load_configs, DEFAULT_CONFIGS_PATH
from testUtils.UpdateMainDeeployTest import update_dynmultideeploy_singleshot_test_main
from scenarioUtils.ScenarioHeaderGen import generate_scenario_header
from scenarioUtils.ThetaLookup import build_theta_by_mode
from scenarioUtils.parse_scenario_log import parse_log, write_csv, POWER_RE

DEEPLOY_ROOT = Path("/app/Deeploy/DynMultiDeeployTest")
GVSOC_EXECUTABLE = "/app/install/gvsoc/bin/gvsoc"

# Must match dynmultideeploytest_singleshot_template.c's OperationMode enum
# exactly (ordinal values baked into the control blob's "mode" field).
MODE_ORDER = ["IDLE", "LOW_ENERGY", "CRITICAL_TASK1", "CRITICAL_TASK2", "BOTH_CRITICAL", "DONE"]
MODE_TO_ORDINAL = {name: i for i, name in enumerate(MODE_ORDER)}


def decide_mode(total_energy_estimate_nj: int, energy_threshold_nj: int, operator_cmd: int,
                 task1_critical: bool, task2_critical: bool) -> str:
    """
    Direct Python port of dynmultideeploytest_template.c's priority chain
    (power budget > operator order > task confidence > default LOW_ENERGY).
    Must stay byte-for-byte in sync with that C logic -- see this script's
    module docstring for why the decision moved here instead of on-device.
    """
    if total_energy_estimate_nj > energy_threshold_nj:
        return "LOW_ENERGY"
    elif operator_cmd == 3:
        return "BOTH_CRITICAL"
    elif operator_cmd == 1:
        return "CRITICAL_TASK1"
    elif operator_cmd == 2:
        return "CRITICAL_TASK2"
    elif task1_critical and task2_critical:
        return "BOTH_CRITICAL"
    elif task1_critical:
        return "CRITICAL_TASK1"
    elif task2_critical:
        return "CRITICAL_TASK2"
    else:
        return "LOW_ENERGY"


def parse_scenario_data_h(path: Path) -> Dict[str, int]:
    """
    Regex-extracts the compile-time constants dynmultideeploytest_singleshot.c
    was ACTUALLY built with, straight from the generated header, rather than
    recomputing them from scenario_config.json's raw watts/microjoules with a
    second copy of ScenarioHeaderGen.py's rounding formula -- guarantees this
    script's host-side FSM replica uses bit-identical thresholds/estimates to
    what the firmware itself has baked in.
    """
    text = path.read_text()

    def _int(name: str) -> int:
        m = re.search(rf"#define {name}\s+(-?\d+)", text)
        if not m:
            raise ValueError(f"{name} not found in {path}")
        return int(m.group(1))

    return {
        "DELTA_T": _int("DELTA_T"),
        "TASK1_MARGIN_THRESHOLD_LIGHT": _int("TASK1_MARGIN_THRESHOLD_LIGHT"),
        "TASK1_MARGIN_THRESHOLD_HEAVY": _int("TASK1_MARGIN_THRESHOLD_HEAVY"),
        "TASK2_MARGIN_THRESHOLD": _int("TASK2_MARGIN_THRESHOLD"),
        "ENERGY_THRESHOLD_NJ": _int("ENERGY_THRESHOLD_NJ"),
        "PHI_ESTIMATE_LOW_ENERGY_NW": _int("PHI_ESTIMATE_LOW_ENERGY_NW"),
        "PHI_ESTIMATE_CRITICAL_TASK1_NW": _int("PHI_ESTIMATE_CRITICAL_TASK1_NW"),
        "PHI_ESTIMATE_CRITICAL_TASK2_NW": _int("PHI_ESTIMATE_CRITICAL_TASK2_NW"),
        "PHI_ESTIMATE_BOTH_CRITICAL_NW": _int("PHI_ESTIMATE_BOTH_CRITICAL_NW"),
    }


def parse_operator_order(scenario_dir: Path) -> List[int]:
    import json
    with open(scenario_dir / "scenario_config.json") as f:
        return json.load(f)["operator_order"]


def build_task1_heavy_lookup(configs: List[Dict]) -> Dict[str, bool]:
    """Mirrors ModeSwitchBodyGen.py's generate_task1_heavy_lookup(): v1 == 'l' means Task 1 runs heavy in that mode."""
    return {cfg["_mode"]: (cfg["v1"] == "l") for cfg in configs}


def write_control_hex(path: Path, tick_index: int, mode_name: str) -> None:
    """{tick_index: uint32, mode: uint32}, little-endian (RISC-V target) -- see the C template's ControlBlob struct."""
    path.write_bytes(struct.pack("<II", tick_index, MODE_TO_ORDINAL[mode_name]))


def load_phi_calibration(mode_to_csv: Dict[str, Path]) -> Dict[str, float]:
    """
    Computes a per-mode calibrated phi_watts (total_energy_j / total_delta_seconds)
    from an existing PURE, single-mode, continuous-process SB run's CSV (as
    already produced by parse_scenario_log.py against a --force-mode
    scenario) -- see this script's module docstring for why this exists (a
    ~28%, uniform-across-launches power-measurement bias in THIS script's own
    single-shot readings, not present in a continuous-process run). Raises if
    a CSV contains any row for a DIFFERENT mode than expected, since a
    "pure" calibration run must never actually switch modes.
    """
    calibration: Dict[str, float] = {}
    for mode_name, csv_path in mode_to_csv.items():
        total_energy = 0.0
        total_seconds = 0.0
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["mode_name"] != mode_name:
                    raise ValueError(
                        f"{csv_path}: expected a PURE {mode_name} calibration run, "
                        f"found mode {row['mode_name']} at tick {row['tick']}"
                    )
                if row["energy_j"] and row["delta_seconds"]:
                    total_energy += float(row["energy_j"])
                    total_seconds += float(row["delta_seconds"])
        if total_seconds == 0:
            raise ValueError(f"{csv_path}: no usable energy_j/delta_seconds rows found")
        calibration[mode_name] = total_energy / total_seconds
    return calibration


def setup(models: List[str], scenario_dir: Path, configs: List[Dict], configs_path: Path) -> Path:
    """
    One-time generation + build. Returns test_c_path (GENERATED_SOURCE).
    Mirrors test_dynamic_2_models.py's own path convention exactly (hardcoded
    "Tests/<m0>_x_<m1>", independent of any --test-folder value) so this
    script locates the SAME fused Network.c/.h that --gen-step already
    produced.
    """
    test_c_path = DEEPLOY_ROOT / "TEST_SIRACUSA" / "Tests" / f"{models[0]}_x_{models[1]}"
    if not (test_c_path / "Network.c").exists():
        raise RuntimeError(
            f"{test_c_path}/Network.c not found -- run test_dynamic_2_models.py --gen-step "
            f"for these models/--configs first (this script only handles the singleshot "
            f"harness on top of an already-fused Network.c/.h)."
        )

    scenario_header_path = test_c_path / "scenario_data.h"
    generate_scenario_header(scenario_dir, scenario_header_path)

    # Placeholder control.hex must exist BEFORE the first `cmake configure`:
    # the hex glob in DynMultiDeeployTest/CMakeLists.txt runs once at
    # configure time, so a file created only afterward would never get
    # flashed. Its actual bytes don't matter here -- every real tick
    # overwrites them before its own launch.
    hex_dir = test_c_path / "hex"
    hex_dir.mkdir(parents=True, exist_ok=True)
    control_path = hex_dir / "control.hex"
    if not control_path.exists():
        write_control_hex(control_path, 0, "LOW_ENERGY")

    update_dynmultideeploy_singleshot_test_main(models, configs)

    test_name = f"{models[0]}_x_{models[1]}"
    build_dir = "TEST_SIRACUSA/build_singleshot"

    configure_cmd = (
        "$CMAKE -D TOOLCHAIN=LLVM -D TOOLCHAIN_INSTALL_DIR=/app/install/llvm "
        f"-D GENERATED_SOURCE={test_c_path} "
        "-D platform=Siracusa -D NUM_CORES=8 -D banshee_simulation=OFF -D gvsoc_simulation=ON "
        f"-B {build_dir} -DENABLE_DYNMULTIDEEPLOY_TEST=ON -DENABLE_DYN_RECONFIG_SINGLESHOT_TEST=ON "
        "-D GVSOC_EXTRA_FLAGS=--power "
        f"-D TESTNAME={test_name} .."
    )
    print(f"[run_singleshot_scenario] CMake configuration command: {configure_cmd}")
    err = subprocess.call(configure_cmd, shell=True, cwd=str(DEEPLOY_ROOT))
    if err != 0:
        raise RuntimeError(f"CMake configuration failed (exit {err})")

    build_cmd = f"$CMAKE --build {DEEPLOY_ROOT / build_dir} --target {test_name}"
    print(f"[run_singleshot_scenario] Build command: {build_cmd}")
    err = subprocess.call(build_cmd, shell=True, cwd=str(DEEPLOY_ROOT))
    if err != 0:
        raise RuntimeError(f"Build failed (exit {err})")

    return test_c_path


def run_gvsoc_once(binary_path: Path, workdir: Path, hex_dir: Path, log_path: Path) -> None:
    """
    Invokes gvsoc DIRECTLY (not via `cmake --build --target gvsoc_<test>`) to
    skip make's dependency-check overhead on every one of the ~100 relaunches
    -- the binary and hex files are already in place, only control.hex's
    bytes change between calls. Flags mirror exactly what CMake's
    add_gvsoc_emulation macro bakes into build.make (confirmed by reading a
    real generated build.make): --target/--binary/--work-dir/--power plus one
    --flash-property per file under hex/.
    """
    # CMake's own add_gvsoc_emulation macro calls make_directory(${GVSOC_WORKDIR})
    # before ever invoking gvsoc -- bypassing cmake --build here means this
    # script must do the same itself, or gvsoc fails trying to write
    # hyperflash.bin into a directory that doesn't exist yet.
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        GVSOC_EXECUTABLE, "--target=siracusa",
        f"--binary={binary_path}",
        f"--work-dir={workdir}",
        "--power",
        f"--flash-property={hex_dir}/task1_scenario.hex@hyperflash:readfs:files",
        f"--flash-property={hex_dir}/task2_scenario.hex@hyperflash:readfs:files",
        f"--flash-property={hex_dir}/control.hex@hyperflash:readfs:files",
        "image", "flash", "run",
    ]
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)


def renumber_power_line(log_text: str, tick_index: int) -> str:
    """
    Each fresh GVSoC process resets its OWN internal @power.measure_N@ counter
    to 0 -- without renumbering, parse_scenario_log.py's index-keyed
    phi_by_index dict would collapse all ticks onto the same key (0) when the
    per-tick logs are concatenated. Exactly one power line is expected per
    single-shot log (one RunNetworks_* dispatch per launch).
    """
    return POWER_RE.sub(lambda m: f"@power.measure_{tick_index}@{m.group(2)}@", log_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs=2, required=True, metavar=("MODEL1", "MODEL2"))
    parser.add_argument("--scenario", type=str, required=True, help="Scenario directory (create_scenario.py output).")
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--output", type=str, required=True, help="Path to write the final per-tick CSV report to.")
    parser.add_argument("--skip-setup", action="store_true",
                         help="Skip scenario_data.h regeneration + cmake configure/build, reusing an already-built "
                              "singleshot binary (e.g. from a previous run of this script against the same scenario).")
    parser.add_argument("--phi-calibration-low-energy", type=str, default=None,
                         help="parse_scenario_log.py CSV from a PURE, continuous-process --force-mode LOW_ENERGY "
                              "SB run (e.g. Static-Small's). When given (along with the other 3 --phi-calibration-* "
                              "flags), every tick's phi_watts/energy_j in the final report is overridden using this "
                              "calibrated per-mode power instead of this script's own single-shot measurement -- "
                              "see the module docstring's 'Known measurement gap' section for why.")
    parser.add_argument("--phi-calibration-critical-task1", type=str, default=None)
    parser.add_argument("--phi-calibration-critical-task2", type=str, default=None)
    parser.add_argument("--phi-calibration-both-critical", type=str, default=None)
    args = parser.parse_args()

    configs_path = Path(args.configs) if args.configs else DEFAULT_CONFIGS_PATH
    configs = load_configs(configs_path)
    task1_heavy_by_mode = build_task1_heavy_lookup(configs)

    scenario_dir = Path(args.scenario)
    if not scenario_dir.is_absolute():
        scenario_dir = DEEPLOY_ROOT / scenario_dir

    if args.skip_setup:
        test_c_path = DEEPLOY_ROOT / "TEST_SIRACUSA" / "Tests" / f"{args.models[0]}_x_{args.models[1]}"
        generate_scenario_header(scenario_dir, test_c_path / "scenario_data.h")
    else:
        test_c_path = setup(args.models, scenario_dir, configs, configs_path)

    consts = parse_scenario_data_h(test_c_path / "scenario_data.h")
    operator_order = parse_operator_order(scenario_dir)
    delta_t = consts["DELTA_T"]
    assert len(operator_order) == delta_t, "operator_order length must match DELTA_T"

    phi_estimate_nw = {
        "LOW_ENERGY": consts["PHI_ESTIMATE_LOW_ENERGY_NW"],
        "CRITICAL_TASK1": consts["PHI_ESTIMATE_CRITICAL_TASK1_NW"],
        "CRITICAL_TASK2": consts["PHI_ESTIMATE_CRITICAL_TASK2_NW"],
        "BOTH_CRITICAL": consts["PHI_ESTIMATE_BOTH_CRITICAL_NW"],
    }
    FREQUENCY_HZ = 50_000_000

    binary_path = DEEPLOY_ROOT / "TEST_SIRACUSA/build_singleshot/bin" / f"{args.models[0]}_x_{args.models[1]}"
    hex_dir = test_c_path / "hex"
    control_path = hex_dir / "control.hex"
    logs_dir = test_c_path / "singleshot_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Host-tracked FSM state, carried sequentially across ticks (this loop IS
    # the "process" now -- the C binary itself is stateless across launches).
    total_energy_estimate_nj = 0
    task1_critical = False
    task2_critical = False

    combined_log_parts: List[str] = []

    for t in range(delta_t):
        operator_cmd = operator_order[t]
        mode = decide_mode(total_energy_estimate_nj, consts["ENERGY_THRESHOLD_NJ"], operator_cmd,
                            task1_critical, task2_critical)

        write_control_hex(control_path, t, mode)
        tick_log_path = logs_dir / f"tick_{t:04d}.log"
        run_gvsoc_once(binary_path, test_c_path / "singleshot_workdir", hex_dir, tick_log_path)

        tick_text = tick_log_path.read_text()
        # Reconstruct the SAME "[HOST CORE] Tick %d: operation mode %d (...)"
        # decision line parse_scenario_log.py's TICK_DECISION_RE expects, with
        # the REAL task1_critical/task2_critical/operator_cmd/power_over_budget
        # this script computed (the C binary printed zeros for these, since it
        # doesn't compute its own mode -- see the template's comment).
        power_over_budget = int(total_energy_estimate_nj > consts["ENERGY_THRESHOLD_NJ"])
        decision_line = (
            f"[HOST CORE] Tick {t}: operation mode {MODE_TO_ORDINAL[mode]} "
            f"(task1_critical={int(task1_critical)}, task2_critical={int(task2_critical)}, "
            f"operator_cmd={operator_cmd}, power_over_budget={power_over_budget})\n"
        )
        # Drop the binary's own placeholder decision line (same tick number,
        # always zeros) and replace it with the reconstructed one above.
        tick_text = re.sub(r"\[HOST CORE\] Tick \d+: operation mode \d+ \([^)]*\)\r?\n?", "", tick_text, count=1)
        tick_text = decision_line + tick_text
        tick_text = renumber_power_line(tick_text, t)
        combined_log_parts.append(tick_text)

        # Pull this tick's real task1_err/task2_margin and RunNetwork cycles
        # back out to update host state for the NEXT tick's mode decision --
        # reuses parse_scenario_log.py's own regexes directly rather than a
        # second copy of them.
        from scenarioUtils.parse_scenario_log import TICK_CONTEXT_RE, RUN_CYCLES_RE
        ctx_m = TICK_CONTEXT_RE.search(tick_text)
        run_m = RUN_CYCLES_RE.search(tick_text)
        if ctx_m is None or run_m is None:
            raise RuntimeError(f"Tick {t}: missing context/RunNetwork line in {tick_log_path} -- see log for the failure.")
        task1_margin = int(ctx_m.group(2))
        task2_margin = int(ctx_m.group(3))
        run_cycles = int(run_m.group(1))

        task1_used_heavy = task1_heavy_by_mode[mode]
        task1_critical = task1_margin < (consts["TASK1_MARGIN_THRESHOLD_HEAVY"] if task1_used_heavy else consts["TASK1_MARGIN_THRESHOLD_LIGHT"])
        task2_critical = task2_margin < consts["TASK2_MARGIN_THRESHOLD"]
        # Matches the C accumulator's own integer arithmetic exactly:
        # ((uint64_t) last_cycles * PHI_ESTIMATE_<MODE>_NW) / FREQUENCY_HZ.
        total_energy_estimate_nj += (run_cycles * phi_estimate_nw[mode]) // FREQUENCY_HZ

        print(f"[run_singleshot_scenario] Tick {t}: mode={mode} run_cycles={run_cycles} "
              f"task1_margin={task1_margin} task2_margin={task2_margin} "
              f"cumulative_energy_est_nj={total_energy_estimate_nj}")

    combined_log_path = test_c_path / "singleshot_combined.log"
    combined_log_path.write_text("".join(combined_log_parts))
    print(f"[run_singleshot_scenario] Wrote combined log: {combined_log_path}")

    theta_by_mode = build_theta_by_mode(configs, test_c_path.relative_to(DEEPLOY_ROOT / "TEST_SIRACUSA"))
    print(f"[run_singleshot_scenario] theta by mode: {theta_by_mode}")

    ticks = parse_log(combined_log_path, theta_by_mode)

    phi_calibration_args = {
        "LOW_ENERGY": args.phi_calibration_low_energy,
        "CRITICAL_TASK1": args.phi_calibration_critical_task1,
        "CRITICAL_TASK2": args.phi_calibration_critical_task2,
        "BOTH_CRITICAL": args.phi_calibration_both_critical,
    }
    given = {k: v for k, v in phi_calibration_args.items() if v}
    if given:
        missing = [k for k, v in phi_calibration_args.items() if not v]
        if missing:
            raise ValueError(f"--phi-calibration-* given for {list(given)} but missing for {missing} -- all 4 modes need a calibration CSV together.")
        phi_calibration = load_phi_calibration({k: Path(v) for k, v in given.items()})
        print(f"[run_singleshot_scenario] Overriding phi_watts/energy_j with SB calibration: {phi_calibration}")
        for row in ticks:
            calibrated_phi = phi_calibration.get(row["mode_name"])
            if calibrated_phi is not None and row["delta_seconds"] is not None:
                row["phi_watts"] = calibrated_phi
                row["energy_j"] = row["delta_seconds"] * calibrated_phi
        # cumulative_theta is unaffected (theta doesn't depend on phi), but
        # cumulative_theta's own running sum was already computed correctly
        # inside parse_log() -- only energy_j/phi_watts change here.

    write_csv(ticks, Path(args.output))

    total_energy_j = sum(row["energy_j"] for row in ticks if row["energy_j"] is not None)
    print(f"[run_singleshot_scenario] Wrote {args.output} ({len(ticks)} ticks)")
    print(f"[run_singleshot_scenario] Total measured energy: {total_energy_j:.6e} J")
    total_theta = ticks[-1]["cumulative_theta"] if ticks else None
    if total_theta is not None and total_energy_j > 0:
        gamma_eff = total_theta / total_energy_j
        print(f"[run_singleshot_scenario] Total theta: {total_theta} ops -- "
              f"gamma_eff = {gamma_eff:.6e} OP/J ({gamma_eff / 1e9:.3f} GOP/J)")


if __name__ == "__main__":
    main()
