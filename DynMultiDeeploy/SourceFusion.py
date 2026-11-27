from typing import Dict, List, Tuple
from pathlib import Path
import re

from DynMultiDeeploy.HeaderFusion import strip_variant_core_suffix

# -----------------------------------------------------------------------------------------#
#                         Functions for fusing multiple Network.c                          #
# -----------------------------------------------------------------------------------------#

GLOBALS_FROM_MARKER = "// ===== GLOBALS from"
# MultiDeeploy's independent-execution redesign no longer emits a "CORES MAP"
# section (each config's combined closure bakes its own core-range boundaries in
# as literal constants -- see MultiDeeploy/CodeFusion/SourceFusion.py's module
# docstring), so the globals section now runs up to the SUBGROUP BARRIERS section
# instead.
SUBGROUP_BARRIERS_START = "// ===== SUBGROUP BARRIERS START ====="
FUSED_FUNCTIONS_MARKER = "// ===== FUSED FUNCTIONS ====="
RUN_NETWORK_MARKER = "void RunNetwork() {"
INIT_NETWORK_MARKER = "void InitNetwork() {"

GLOBALS_START = "// ===== GLOBALS START ====="
GLOBALS_END = "// ===== GLOBALS END ====="
FUNCTIONS_START = "// ===== FUNCTIONS START ====="
FUNCTIONS_END = "// ===== FUNCTIONS END ====="
RUNNETWORKS_START = "// ===== RUNNETWORKS START ====="
RUNNETWORKS_END = "// ===== RUNNETWORKS END ====="
INITNETWORKS_START = "// ===== INITNETWORKS START ====="
INITNETWORKS_END = "// ===== INITNETWORKS END ====="

PI_L2_NAME_RE = re.compile(r"static\s+PI_L2\s+\S+\s+(\w+)")


def split_statements(text: str) -> List[str]:
    """
    Splits text into a list of individual C statements, keeping brace-enclosed
    initializers that span multiple lines intact. Comment-only, blank and
    preprocessor-directive (#include/#define/...) lines are kept as their own
    separate entries, so that they never get prepended to the statement that
    follows them (those lines have no terminating ';', so without this they'd
    otherwise merge with whatever real statement comes next).
    """
    statements: List[str] = []
    current: List[str] = []
    depth = 0

    for line in text.splitlines(keepends = True):
        stripped = line.strip()

        if depth == 0 and not current and (stripped == "" or stripped.startswith(("//", "#"))):
            statements.append(line)
            continue

        current.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0 and ";" in line:
            statements.append("".join(current))
            current = []
            depth = 0

    if current:
        statements.append("".join(current))

    return statements


def fuse_globals(files: List[Path], variant_tags: List[str]) -> Tuple[str, Dict[str, str]]:
    """
    Fuses the GLOBALS section of multiple Network.c files.
    Args:
        files: List of paths to Network.c files whose GLOBALS section should be fused.
        variant_tags: Per-(task, variant) tags (e.g. "AnomalyDetection_light",
            "AnomalyDetection_heavy"), used to detect that PI_L2 (weight) variables
            that only differ by their core-count suffix (e.g.
            AnomalyDetection_light_4_x vs AnomalyDetection_light_2_x) belong to the
            SAME variant and must only be emitted once, while variables belonging to a
            DIFFERENT variant of the same task (e.g. AnomalyDetection_heavy_*) are
            never merged with it. MEMORYARENA/input/output/PI_L1 variables are
            core-split specific (their size can differ per configuration) and are
            always kept as-is, with their suffix untouched.
    Returns:
        A tuple (globals_text, l2_rename_map):
          - globals_text: the fused GLOBALS content as a single string.
          - l2_rename_map: maps every PI_L2 variable name seen (kept or dropped as a
            duplicate) to its canonical, suffix-stripped name. Callers must apply this
            renaming to the rest of the fused Network.c so that occurrences of dropped
            duplicates point to the single retained declaration.
    """
    fused_statements: List[str] = []
    seen_l2_vars = set()
    l2_rename_map: Dict[str, str] = {}

    for path in files:
        text = path.read_text()
        globals_text = text[text.index(GLOBALS_FROM_MARKER):text.index(SUBGROUP_BARRIERS_START)]

        for statement in split_statements(globals_text):
            stripped = statement.strip()

            if stripped.startswith(("#include", "#define")):
                continue

            if stripped.startswith("static PI_L2"):
                name = PI_L2_NAME_RE.match(stripped).group(1)
                canonical_name = strip_variant_core_suffix(name, variant_tags)
                l2_rename_map[name] = canonical_name

                if canonical_name in seen_l2_vars:
                    continue
                seen_l2_vars.add(canonical_name)
                statement = re.sub(rf"\b{re.escape(name)}\b", canonical_name, statement)

            fused_statements.append(statement)

    return "".join(fused_statements), l2_rename_map


def split_Network_c(text: str) -> Tuple[str, str, str]:
    """
    Splits a Network.c file into 3 sections:
      1) functions: from the first 'FUSED FUNCTIONS: Scheduling Round 0' marker,
         up to (excluding) 'void RunNetwork() {'
      2) run_network: from 'void RunNetwork() {' (included),
         up to (excluding) 'void InitNetwork() {'
      3) init_network: from 'void InitNetwork() {' (included) to the end of the file
    """
    functions_idx = text.index(FUSED_FUNCTIONS_MARKER)
    run_idx = text.index(RUN_NETWORK_MARKER)
    init_idx = text.index(INIT_NETWORK_MARKER)

    functions = text[functions_idx:run_idx]
    run_network = text[run_idx:init_idx]
    init_network = text[init_idx:]

    return functions, run_network, init_network


def rename_function(text: str, old_name: str, new_name: str) -> str:
    """Renames every whole-word occurrence of old_name to new_name."""
    return re.sub(rf"\b{re.escape(old_name)}\b", new_name, text)


COMBINED_CLOSURE_RE = re.compile(r"\b(Combined_closure)_(\d+_\d+)(_args_t)?\b")


def disambiguate_round_closures(text: str, v1: str, v2: str) -> str:
    """
    Inserts the (v1, v2) variant infix into the config's combined closure name and
    its _args_t struct (e.g. Combined_closure_4_4 -> Combined_closure_s_s_4_4,
    Combined_closure_4_4_args_t -> Combined_closure_s_s_4_4_args_t).
    MultiDeeploy.CodeFusion.SourceFusion names these purely from the (cores_0,
    cores_1) split, with no model or variant information, so two configs that
    happen to share a core split (e.g. LE and CC both at 4/4, despite running
    different variants) would otherwise emit two C functions/structs with the
    exact same name once fused into one file. The optional "_args_t" suffix must
    be matched as part of the same pattern rather than relying on a trailing \b: a
    digit is immediately followed by "_", and both are word characters, so \b
    never falls between them -- a separate \b-anchored pass would silently skip
    every "_args_t" struct name. Must be applied per-config, before
    concatenating configs together, since the closure names it targets always carry
    that config's own core split already.
    """
    def _replace(m: "re.Match[str]") -> str:
        prefix, cores, args_t_suffix = m.group(1), m.group(2), m.group(3) or ""
        return f"{prefix}_{v1}_{v2}_{cores}{args_t_suffix}"

    return COMBINED_CLOSURE_RE.sub(_replace, text)


def insert_section(text: str, start_marker: str, end_marker: str, content: str) -> str:
    """Replaces the (possibly empty) content between start_marker and end_marker, both kept, with content."""
    pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)
    replacement = f"{start_marker}\n{content}\n{end_marker}"
    return pattern.sub(lambda _: replacement, text, count = 1)


def fuse_network_c(
    template_path: Path,
    files: List[Path],
    core_splits: List[Tuple[int, int]],
    variant_names: List[Tuple[str, str]],
    variant_tags: List[str]
) -> str:
    """
    Fuses the GLOBALS, FUNCTIONS, RUNNETWORKS and INITNETWORKS sections of multiple
    Network.c files into the Network.c template.
    Args:
        template_path: Path to the Network.c template (DynMultiDeeploy/Templates/Network.c).
        files: List of paths to Network.c files to be fused, one per hardware configuration.
        core_splits: One (cores_task_0, cores_task_1) tuple per file, in the same order as
            `files`, used to build part of the RunNetworks/InitNetworks suffix (e.g. "_2_6").
        variant_names: One (variant_task_0, variant_task_1) tuple per file, in the same
            order as `files`, e.g. ("light", "l") -- the per-task variant label used to
            build the required `RunNetworks_v1_v2_n1_n2` / `InitNetworks_v1_v2_n1_n2`
            naming (e.g. "RunNetworks_s_s_4_4"), so that configs sharing the same core
            split but different variants (e.g. LE and CC both at 4/4) get distinct
            entry points.
        variant_tags: Per-(task, variant) tags (e.g. "AnomalyDetection_light",
            "AnomalyDetection_heavy"), used to dedupe the GLOBALS PI_L2 (weight)
            variables that only differ by their core-count suffix, without conflating
            different variants of the same task.
    Returns:
        The template filled with the fused GLOBALS, FUNCTIONS, RUNNETWORKS and
        INITNETWORKS sections. Only the deduped PI_L2 (weight) variables are renamed to
        their canonical, suffix-stripped name; MEMORYARENA/input/output/PI_L1 variables,
        which are core-split specific, keep their original per-configuration name.
    """
    assert len(files) == len(core_splits), "One core split per Network.c file is required."
    assert len(files) == len(variant_names), "One variant pair per Network.c file is required."

    functions_parts: List[str] = []
    run_networks_parts: List[str] = []
    init_networks_parts: List[str] = []

    for path, (g0, g1), (v1, v2) in zip(files, core_splits, variant_names):
        text = path.read_text()
        text = disambiguate_round_closures(text, v1, v2)
        functions, run_network, init_network = split_Network_c(text)

        suffix = f"_{v1}_{v2}_{g0}_{g1}"
        run_network = rename_function(run_network, "RunNetwork", f"RunNetworks{suffix}")
        init_network = rename_function(init_network, "InitNetwork", f"InitNetworks{suffix}")

        functions_parts.append(functions)
        run_networks_parts.append(run_network)
        init_networks_parts.append(init_network)

    globals_text, l2_rename_map = fuse_globals(files, variant_tags)

    template = template_path.read_text()
    template = insert_section(template, GLOBALS_START, GLOBALS_END, globals_text)
    template = insert_section(template, FUNCTIONS_START, FUNCTIONS_END, "\n".join(functions_parts))
    template = insert_section(template, RUNNETWORKS_START, RUNNETWORKS_END, "\n".join(run_networks_parts))
    template = insert_section(template, INITNETWORKS_START, INITNETWORKS_END, "\n".join(init_networks_parts))

    for name, canonical_name in l2_rename_map.items():
        if name != canonical_name:
            template = rename_function(template, name, canonical_name)

    return template
