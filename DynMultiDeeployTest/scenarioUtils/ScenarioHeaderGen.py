#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Converts a scenario directory (scenario_config.json + task{1,2}_inputs.npz, as
produced by create_scenario.py) into a single C header (scenario_data.h) that the
DynSiracusa main-loop FSM includes directly -- the operator's-order timeline and the
threshold/power-estimate constants -- plus two raw binary blobs
(<output_path.parent>/hex/task{1,2}_scenario.hex) holding the per-tick input rows.

The per-tick inputs are NOT embedded as `static const` C arrays anymore: at
DELTA_T=30+ the combined Task1+Task2 arrays, resident in L2 alongside the
always-resident weight globals for all 4 variants, exhausted L2 and made
InitNetworks_*'s pi_l2_malloc/pmsis_l1_malloc calls for the runtime arena fail
silently (observed as an "Invalid access" at the poisoned address 0x57575757).
Instead, the two blobs are flashed into the simulated HyperFlash image (via the
existing generic `file(GLOB_RECURSE HEXLIST "${GENERATED_SOURCE}/hex/**")` +
`gvsoc_flags_add_files_to_hyperflash` CMake mechanism, already wired in
DynMultiDeeployTest/CMakeLists.txt for exactly this purpose -- see
Deeploy/Targets/PULPOpen/Deployer.py's `_L3AllocTemplate`/`_L3InitTemplate` for the
same pattern used for oversized weight tensors) and streamed into HyperRAM once at
boot (`load_file_to_ram`), then DMA'd a tick-row at a time into a small,
DELTA_T-independent L2 staging buffer (`ram_read`) by the main-loop FSM. This makes
the static L2 footprint O(1) in DELTA_T instead of O(DELTA_T).
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

def _flatten_ticks(arr: np.ndarray) -> np.ndarray:
    """(T, *sample_shape) -> (T, prod(sample_shape)), clipped to int8 range."""
    delta_t = arr.shape[0]
    flat = arr.reshape(delta_t, -1).astype(np.int64)
    return np.clip(flat, -128, 127)


TASK1_HEX_FILENAME = "task1_scenario.hex"
TASK2_HEX_FILENAME = "task2_scenario.hex"


def _write_hex_blob(hex_dir: Path, filename: str, values: np.ndarray) -> None:
    """
    Writes a flat, row-major (tick t's data at [t*width, (t+1)*width)) raw int8 byte
    blob -- loaded whole into HyperRAM at boot via load_file_to_ram(), then streamed a
    tick-row at a time into a small static L2 buffer via ram_read(). Filename must end
    in ".hex" to match load_file_to_ram()'s hardcoded "${extName}.hex" convention
    (despite the name, this is a plain binary blob, not Intel-hex text -- readfs/
    load_file_to_ram never parse it as anything but raw bytes).
    """
    hex_dir.mkdir(parents = True, exist_ok = True)
    (hex_dir / filename).write_bytes(values.reshape(-1).astype(np.int8).tobytes())


def generate_scenario_header(scenario_dir: Path, output_path: Path) -> None:
    with open(scenario_dir / "scenario_config.json") as f:
        config = json.load(f)

    task1_npz = np.load(scenario_dir / "task1_inputs.npz")["inputs"]
    task2_npz = np.load(scenario_dir / "task2_inputs.npz")["inputs"]

    task1_flat = _flatten_ticks(task1_npz)
    task2_flat = _flatten_ticks(task2_npz)

    delta_t = config["delta_t"]
    assert task1_flat.shape[0] == delta_t and task2_flat.shape[0] == delta_t, \
        "scenario_config.json's delta_t must match the number of generated input samples"

    thresholds = config["thresholds"]
    phi = config["phi_estimate_watts"]
    operator_order = config["operator_order"]

    # Integer-only on the target: the fabric controller (where main()/the FSM runs)
    # crashed GVSoC outright on ANY floating-point arithmetic (confirmed by isolating
    # down to a bare `float x; x += 1.0f;` in the main loop -- the FC most likely has
    # no hardware FPU, only the cluster cores do, and GVSoC doesn't model the
    # resulting illegal-instruction trap gracefully). Power estimates are therefore
    # emitted in nanowatts and energy in nanojoules, both plain integers: energy[nJ]
    # = power[nW] * cycles / freq[Hz] is dimensionally exact in integer arithmetic
    # (nW = nJ/s, so nW * (cycles/Hz) = nW * s = nJ).
    def _to_nanowatts(watts: float) -> int:
        return int(round(watts * 1e9))

    lines: List[str] = [
        "#ifndef __SCENARIO_DATA_H__",
        "#define __SCENARIO_DATA_H__",
        "",
        f"#define DELTA_T {delta_t}",
        f"#define TASK1_INPUT_SIZE {task1_flat.shape[1]}",
        f"#define TASK2_INPUT_SIZE {task2_flat.shape[1]}",
        "",
        f"#define TASK1_MARGIN_THRESHOLD_LIGHT {int(thresholds['task1_margin_threshold_light'])}",
        f"#define TASK1_MARGIN_THRESHOLD_HEAVY {int(thresholds['task1_margin_threshold_heavy'])}",
        f"#define TASK2_MARGIN_THRESHOLD {int(thresholds['task2_margin_threshold'])}",
        f"#define ENERGY_THRESHOLD_NJ {int(round(float(thresholds['energy_threshold_uj']) * 1000.0))}ULL",
        "",
        f"#define PHI_ESTIMATE_LOW_ENERGY_NW {_to_nanowatts(phi['LOW_ENERGY'])}ULL",
        f"#define PHI_ESTIMATE_CRITICAL_TASK1_NW {_to_nanowatts(phi['CRITICAL_TASK1'])}ULL",
        f"#define PHI_ESTIMATE_CRITICAL_TASK2_NW {_to_nanowatts(phi['CRITICAL_TASK2'])}ULL",
        f"#define PHI_ESTIMATE_BOTH_CRITICAL_NW {_to_nanowatts(phi['BOTH_CRITICAL'])}ULL",
        "",
        f'#define TASK1_SCENARIO_HEX_FILE "{TASK1_HEX_FILENAME}"',
        f'#define TASK2_SCENARIO_HEX_FILE "{TASK2_HEX_FILENAME}"',
        "",
        # Tiny (DELTA_T bytes) regardless of scenario size -- kept as a static L2
        # array, unlike the per-tick input rows above.
        f"static const uint8_t operator_order[DELTA_T] = {{{', '.join(str(v) for v in operator_order)}}};",
        "",
        "#endif",
        "",
    ]

    output_path.write_text("\n".join(lines))

    hex_dir = output_path.parent / "hex"
    _write_hex_blob(hex_dir, TASK1_HEX_FILENAME, task1_flat)
    _write_hex_blob(hex_dir, TASK2_HEX_FILENAME, task2_flat)

    print(f"[ScenarioHeaderGen] Wrote {output_path} (DELTA_T={delta_t}, "
          f"TASK1_INPUT_SIZE={task1_flat.shape[1]}, TASK2_INPUT_SIZE={task2_flat.shape[1]})")
    print(f"[ScenarioHeaderGen] Wrote {hex_dir}/{TASK1_HEX_FILENAME} "
          f"({task1_flat.nbytes} bytes) and {hex_dir}/{TASK2_HEX_FILENAME} ({task2_flat.nbytes} bytes)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate scenario_data.h from a scenario directory.")
    parser.add_argument("--scenario-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    generate_scenario_header(Path(args.scenario_dir), Path(args.output))
