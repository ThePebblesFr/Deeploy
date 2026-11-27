#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Per-mode theta (total ops for one co-resident inference of that mode's task1+task2
pair), read from the profiling JSONs testRunner_2_models.py already writes
unconditionally (--profileTiling --profileToJSON, on every per-config sub-model) --
no new generation step needed, just reading files that are already on disk once
--gen-step has run.

Path convention matches testRunner_2_models.py exactly (`test_dir` is the same
--test-folder/-t value passed there and to test_dynamic_2_models.py, e.g.
"Tests/TinyClassifier_x_AnomalyDetection", WITHOUT a "TEST_SIRACUSA/" prefix --
testRunner_2_models.py prepends that itself, so this does too):
TEST_SIRACUSA/<test_dir>/<variant0>_<cores0>_x_<variant1>_<cores1>/<variant0>_<cores0>/
deeployStates/<variant0>_<cores0>_profiling.json (and the same shape for the
variant1/cores1 side).
"""

import json
from pathlib import Path
from typing import Dict, List

from MultiDeeploy.ModelProfile import ModelProfile


def _model_tag(variant: str, cores: int) -> str:
    return f"{variant}_{cores}"


def _profiling_json_path(test_dir: Path, cfg: Dict, model_tag: str) -> Path:
    m0 = _model_tag(cfg["variant0"], cfg["cores0"])
    m1 = _model_tag(cfg["variant1"], cfg["cores1"])
    return Path("TEST_SIRACUSA") / test_dir / f"{m0}_x_{m1}" / model_tag / "deeployStates" / f"{model_tag}_profiling.json"


def _load_total_ops(profiling_json_path: Path) -> int:
    if not profiling_json_path.exists():
        raise FileNotFoundError(
            f"{profiling_json_path} not found -- run test_dynamic_2_models.py with --gen-step first "
            f"(testRunner_2_models.py writes this unconditionally as part of that step)."
        )
    with open(profiling_json_path) as f:
        profile = ModelProfile.from_dict(json.load(f))
    total_ops = profile.total_ops()
    if total_ops is None:
        raise ValueError(
            f"{profiling_json_path} has no per-layer 'ops' field -- it was generated before "
            f"testRunner.py started capturing it. Rerun --gen-step to regenerate it."
        )
    return total_ops


def build_theta_by_mode(configs: List[Dict], test_dir: Path) -> Dict[str, int]:
    """
    Returns {mode_name: theta}, theta = task2's total_ops + task1's total_ops for
    that mode's specific (variant, cores) pair.
    """
    theta_by_mode: Dict[str, int] = {}
    for cfg in configs:
        m0 = _model_tag(cfg["variant0"], cfg["cores0"])
        m1 = _model_tag(cfg["variant1"], cfg["cores1"])
        theta_task2 = _load_total_ops(_profiling_json_path(test_dir, cfg, m0))
        theta_task1 = _load_total_ops(_profiling_json_path(test_dir, cfg, m1))
        theta_by_mode[cfg["_mode"]] = theta_task2 + theta_task1
    return theta_by_mode
