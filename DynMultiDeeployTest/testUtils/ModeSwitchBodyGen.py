import re
from typing import Dict, List

"""
Generates the per-mode switch bodies in dynmultideeploytest_template.c from the
config list (DynMultiDeeployTest/Configs/*.json, see ConfigLoader.py) instead of
them being hand-written -- the last hand-maintained piece Phase 2 explicitly called
out as still needing this. The FSM's mode SET and decision logic (4 fixed modes,
power > operator-order > task-confidence priority) stay hand-written in the
template: they're paper-intrinsic and don't vary with which model directories a
config plugs into each mode. What's generated is only the MAPPING from mode ->
(model directories, core split, weight-arena symbols, entry-point names) -- the part
that changes whenever --configs points at a different JSON file.

Each config entry must carry "_mode" naming which of the 4 fixed OperationMode
enum values it maps to (see Configs/tinyclassifier_anomalydetection_4mode.json).
variant0/cores0 is always the Task 2 (models_name[0]) side, variant1/cores1 always
Task 1 (models_name[1]) -- this fixed convention is unchanged from Phase 1/2, not
something this generator introduces.

Assumes (matching every hand-written mode body before this generator existed, not a
new limitation): each model has exactly one input tensor buffer per num_inputs slot
copied from that task's flat tick buffer, and exactly one output tensor (`_output_0`).
"""

OPERATION_MODE_ORDER = ["IDLE", "LOW_ENERGY", "CRITICAL_TASK1", "CRITICAL_TASK2", "BOTH_CRITICAL", "DONE"]

MARKERS = {
    "TASK1_HEAVY_LOOKUP": ("// ===== GENERATED: TASK1_HEAVY_LOOKUP START =====",
                           "// ===== GENERATED: TASK1_HEAVY_LOOKUP END ====="),
    "ARENA_FREE_CASES": ("// ===== GENERATED: ARENA_FREE_CASES START =====",
                         "// ===== GENERATED: ARENA_FREE_CASES END ====="),
    "MODE_DISPATCH_CASES": ("// ===== GENERATED: MODE_DISPATCH_CASES START =====",
                            "// ===== GENERATED: MODE_DISPATCH_CASES END ====="),
}


def _task2_model(cfg: Dict) -> str:
    return f"{cfg['variant0']}_{cfg['cores0']}"


def _task1_model(cfg: Dict) -> str:
    return f"{cfg['variant1']}_{cfg['cores1']}"


def generate_task1_heavy_lookup(configs: List[Dict]) -> str:
    """
    Replaces the old hand-written `(operation_mode == CRITICAL_TASK1 ||
    operation_mode == BOTH_CRITICAL)` guess at which modes run Task 1's heavy
    variant with a lookup driven by each config's actual v1 ('l' = heavy),
    indexed directly by the OperationMode enum value.
    """
    by_mode = {cfg["_mode"]: cfg for cfg in configs}
    values = ["1" if (name in by_mode and by_mode[name]["v1"] == "l") else "0"
              for name in OPERATION_MODE_ORDER]
    return (f"static const int TASK1_MODE_IS_HEAVY[{len(OPERATION_MODE_ORDER)}] = "
            f"{{{', '.join(values)}}};")


def generate_arena_free_cases(configs: List[Dict]) -> str:
    lines: List[str] = []
    for cfg in configs:
        m0, m1 = _task2_model(cfg), _task1_model(cfg)
        lines += [
            f"                case {cfg['_mode']}:",
            f"                    pmsis_l1_malloc_free({m0}_MEMORYARENA_L1, {m0}_MEMORYARENA_L1_len);",
            f"                    pi_l2_free({m0}_MEMORYARENA_L2, {m0}_MEMORYARENA_L2_len);",
            f"                    pmsis_l1_malloc_free({m1}_MEMORYARENA_L1, {m1}_MEMORYARENA_L1_len);",
            f"                    pi_l2_free({m1}_MEMORYARENA_L2, {m1}_MEMORYARENA_L2_len);",
            "                    break;",
        ]
    return "\n".join(lines)


def generate_mode_dispatch_cases(configs: List[Dict]) -> str:
    blocks: List[str] = []
    for cfg in configs:
        mode = cfg["_mode"]
        v0, v1, c0, c1 = cfg["v0"], cfg["v1"], cfg["cores0"], cfg["cores1"]
        m0, m1 = _task2_model(cfg), _task1_model(cfg)
        init_fn = f"InitNetworks_{v0}_{v1}_{c0}_{c1}"
        run_fn = f"RunNetworks_{v0}_{v1}_{c0}_{c1}"
        phi_define = f"PHI_ESTIMATE_{mode}_NW"
        blocks.append(f"""            case {mode}:
                if (mode_changed) {{
                    // send {init_fn}() cluster task
                    printf("[HOST CORE] Sending {init_fn}() cluster task\\r\\n");
                    pi_cluster_task(&cluster_task, {init_fn}, NULL);
                    cluster_task.stack_size = MAINSTACKSIZE;
                    cluster_task.slave_stack_size = SLAVESTACKSIZE;
                    ResetTimer();
                    StartTimer();
                    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                    StopTimer();
                    printf("Runtime InitNetwork: %u cycles\\r\\n", getCycles());
                }}
                // {m0} = Task 2 ({c0} cores)
                for (uint32_t buf = 0; buf < {m0}_num_inputs; buf++) {{
                    if ((uint32_t) {m0}_inputs[buf] >= 0x10000000) {{
                    memcpy({m0}_inputs[buf], task2_tick_buf,
                            {m0}_inputs_bytes[buf]);
                    }}
                }}
                // {m1} = Task 1 ({c1} cores)
                for (uint32_t buf = 0; buf < {m1}_num_inputs; buf++) {{
                    if ((uint32_t) {m1}_inputs[buf] >= 0x10000000) {{
                    memcpy({m1}_inputs[buf], task1_tick_buf,
                            {m1}_inputs_bytes[buf]);
                    }}
                }}
                // send {run_fn}() cluster task
                printf("[HOST CORE] Sending {run_fn}() cluster task\\r\\n");
                pi_cluster_task(&cluster_task, {run_fn}, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                last_cycles = getCycles();
                printf("Runtime RunNetwork: %u cycles\\r\\n", last_cycles);
                task1_marg = compute_margin({m1}_output_0, TASK1_NUM_CLASSES);
                task2_marg = compute_margin({m0}_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * {phi_define}) / FREQUENCY_HZ;
                break;""")
    return "\n".join(blocks)


def inject_generated_sections(content: str, configs: List[Dict]) -> str:
    """
    Replaces the text between each START/END marker pair (markers themselves kept,
    so this is idempotent/re-runnable against either the pristine template or a
    previously-generated dynmultideeploytest.c) with freshly generated code for the
    given config list.
    """
    generated = {
        "TASK1_HEAVY_LOOKUP": generate_task1_heavy_lookup(configs),
        "ARENA_FREE_CASES": generate_arena_free_cases(configs),
        "MODE_DISPATCH_CASES": generate_mode_dispatch_cases(configs),
    }

    for name, (start, end) in MARKERS.items():
        # Indentation is read back from the template's own marker lines rather than
        # assumed here, so this works regardless of which nesting level a given
        # marker pair sits at.
        pattern = re.compile(
            r"(?P<start_ws>[ \t]*)" + re.escape(start) + r".*?"
            r"(?P<end_ws>[ \t]*)" + re.escape(end),
            re.DOTALL,
        )
        m = pattern.search(content)
        if not m:
            raise ValueError(f"Marker pair {name} ({start!r} / {end!r}) not found in template")
        replacement = f"{m.group('start_ws')}{start}\n{generated[name]}\n{m.group('end_ws')}{end}"
        content = content[:m.start()] + replacement + content[m.end():]

    return content
