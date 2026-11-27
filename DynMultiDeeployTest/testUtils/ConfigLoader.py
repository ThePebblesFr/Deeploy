import json
from pathlib import Path
from typing import Dict, List

# Fields every mode-config entry must carry. Keys match what
# DynMultiDeeploy.SourceFusion.fuse_network_c() and the per-config
# testRunner_2_models.py invocation in test_dynamic_2_models.py already expect --
# unchanged from the pre-JSON hardcoded CONFIGS list, so existing config-consuming
# code needed no changes when this loader was introduced.
REQUIRED_CONFIG_KEYS = {"variant0", "variant1", "cores0", "cores1", "l1_0", "l1_1", "v0", "v1"}

DEFAULT_CONFIGS_PATH = Path(__file__).resolve().parent.parent / "Configs" / "tinyclassifier_anomalydetection_4mode.json"


def load_configs(configs_path: Path) -> List[Dict]:
    """
    Loads a mode-config list from a JSON file shaped like
    DynMultiDeeployTest/Configs/tinyclassifier_anomalydetection_4mode.json:
    {"configs": [{"variant0": ..., "variant1": ..., "cores0": ..., "cores1": ...,
    "l1_0": ..., "l1_1": ..., "v0": ..., "v1": ...}, ...]}. Order matters: it must
    match the 4 hardcoded case blocks (LOW_ENERGY, CRITICAL_TASK1, CRITICAL_TASK2,
    BOTH_CRITICAL, in that order) in dynmultideeploytest_template.c -- swapping in a
    config with genuinely different model directories still requires updating that
    template's literal per-mode symbol names to match (Phase 3 will generate them
    from this same file instead).
    """
    if not configs_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {configs_path}. Pass --configs pointing to a JSON file shaped like "
            f"{DEFAULT_CONFIGS_PATH}."
        )

    with open(configs_path) as f:
        data = json.load(f)

    configs = data["configs"] if isinstance(data, dict) else data
    if not isinstance(configs, list) or len(configs) != 4:
        raise ValueError(
            f"{configs_path}: expected a list of exactly 4 mode configs (LOW_ENERGY, CRITICAL_TASK1, "
            f"CRITICAL_TASK2, BOTH_CRITICAL, in that order to match dynmultideeploytest_template.c's "
            f"hardcoded case blocks), got {len(configs) if isinstance(configs, list) else type(configs).__name__}."
        )

    for idx, cfg in enumerate(configs):
        missing = REQUIRED_CONFIG_KEYS - cfg.keys()
        if missing:
            raise ValueError(f"{configs_path}: config entry {idx} is missing required key(s): {sorted(missing)}")

    return configs
