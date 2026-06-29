import re
from collections import defaultdict
from typing import Dict, Tuple, Any, List

def normalize_layer_name(name: str) -> str:
    """
    Normalize layer names coming from:
      - ModelProfile (suffix _L2, _L3, etc.)
      - FunctionCall (contains closure, memory level)
    """
    if "___" in name:
        parts = name.split("___", 1)
        print(parts)

        model, layer = parts
        return f"{model}__{layer}"

    if "__" in name:
        return name  # already normalized

    # Split only on the FIRST underscore
    parts = name.split("_", 1)

    model, layer = parts
    return f"{model}__{layer}"

def split_struct_and_function(code: str) -> tuple[str, str]:
    """
    Splits:
      typedef struct {...} name_t;
      <function definition>

    Returns (struct_part, function_part)
    """

    lines = code.splitlines(keepends=True)

    struct_lines = []
    func_lines = []

    in_struct = True
    brace_depth = 0

    for line in lines:
        if in_struct:
            struct_lines.append(line)
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            if brace_depth == 0 and "}" in line:
                in_struct = False
        else:
            func_lines.append(line)

    return "".join(struct_lines), "".join(func_lines)

def split_typedef_struct(code: str) -> list[str]:
    """
    Split:
      typedef struct {
        ...
      } name_t;

    → [prefix, args, suffix]
    """
    lines = code.splitlines(keepends=True)

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) < 2:
        return ["", "", ""]

    prefix = lines[0]

    end_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "}" in lines[i]:
            end_idx = i
            break

    if end_idx is None:
        raise RuntimeError("Malformed struct")

    args = "".join(lines[1:end_idx])
    suffix = lines[end_idx]

    return [prefix, args, suffix]


def split_function_call(code: str) -> list[str]:
    """
    Split function call block into:
      [prefix, args, suffix]

    prefix → all lines up to and including the first '{'
    args   → same logic as before
    suffix → same logic as before
    """

    lines = code.splitlines(keepends=True)

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ["", "", ""]

    prefix_lines = []
    i = 0
    while i < len(lines):
        prefix_lines.append(lines[i])
        if "{" in lines[i]:
            i += 1
            break
        i += 1
    remaining = lines[i:]

    if len(remaining) == 0:
        return ["".join(prefix_lines), "", ""]

    if len(remaining) == 1:
        return ["".join(prefix_lines), "", remaining[0]]

    return [
        "".join(prefix_lines),
        "".join(remaining[:-1]),
        remaining[-1],
    ]

def split_closure_L3_call_body(block: str) -> dict[str, str]:
    """
    Splits a closure_L3 call body into:
      - args_cast
      - closure_function_call
      - closure_call
    """

    ARG_CAST_START = "// CLOSURE ARG CAST"
    FUNC_CALL_START = "// CLOSURE FUNCTION CALL"
    CALL_START_SUFFIX = " CLOSURE CALL"
    WRITEBACK_START = "// CLOSURE ARG WRITEBACK"

    args_cast_lines = []
    closure_function_call_lines = []
    closure_call_lines = []

    mode = None  # None | "args_cast" | "closure_function" | "closure_call"

    for line in block.splitlines(keepends=True):

        if ARG_CAST_START in line:
            mode = "args_cast"
            continue

        if FUNC_CALL_START in line:
            mode = "closure_function"
            continue

        if CALL_START_SUFFIX in line:
            mode = "closure_call"
            continue

        if WRITEBACK_START in line:
            mode = None
            break

        if mode == "args_cast":
            args_cast_lines.append(line)
        elif mode == "closure_function":
            closure_function_call_lines.append(line)
        elif mode == "closure_call":
            closure_call_lines.append(line)

    return {
        "args_cast": "".join(args_cast_lines),
        "closure_function_call": "".join(closure_function_call_lines),
        "closure_call": "".join(closure_call_lines),
    }

def split_compound_literal(block: str) -> tuple[str, str, str]:
    """
    Splits a C compound literal into:
      - prefix  (up to and including '{')
      - args    (between '{' and '}')
      - suffix  (from '}' onward, included)
    """

    lines = block.splitlines(keepends=True)

    prefix_lines = []
    args_lines = []
    suffix_lines = []

    state = "prefix"  # prefix | args | suffix

    for line in lines:

        if state == "prefix":
            if "{" in line:
                before, after = line.split("{", 1)
                prefix_lines.append(before + "{\n")
                if after.strip():
                    args_lines.append(after)
                state = "args"
            else:
                prefix_lines.append(line)

        elif state == "args":
            if "}" in line:
                before, after = line.split("}", 1)
                if before.strip():
                    args_lines.append(before + "\n")
                suffix_lines.append("}" + after)
                state = "suffix"
            else:
                args_lines.append(line)

        else:  # suffix
            suffix_lines.append(line)

    return (
        "".join(prefix_lines),
        "".join(args_lines),
        "".join(suffix_lines),
    )

def remove_first_lines(code: str) -> str:
    lines = code.splitlines(keepends=True)
    removed_cast = False
    i_removed_cast = 0
    for i, line in enumerate(lines):
        if not removed_cast and ";" in line:
            i_removed_cast = i + 1
            removed_cast = True
    return "".join(lines[i_removed_cast:])

# CLOSURE FUNCTIONS HELPERS

def split_closure_call_body(block: str) -> dict[str, str]:
    """
    For 'closure' call bodies (not closure_L3), split into:
      - arg_cast: between '// CLOSURE ARG CAST' and '// CLOSURE FUNCTION CALL'
      - body:     between '// CLOSURE FUNCTION CALL' and '// CLOSURE ARG WRITEBACK'
    """
    ARG_CAST_START = "// CLOSURE ARG CAST"
    FUNC_CALL_START = "// CLOSURE FUNCTION CALL"
    WRITEBACK_START = "// CLOSURE ARG WRITEBACK"

    arg_cast_lines: list[str] = []
    body_lines: list[str] = []

    mode = None  # None | "arg_cast" | "body"

    for line in block.splitlines(keepends=True):
        if ARG_CAST_START in line:
            mode = "arg_cast"
            continue
        if FUNC_CALL_START in line:
            mode = "body"
            continue
        if WRITEBACK_START in line:
            mode = None
            break

        if mode == "arg_cast":
            arg_cast_lines.append(line)
        elif mode == "body":
            body_lines.append(line)

    arg_cast_lines = remove_first_lines("".join(arg_cast_lines)).splitlines(keepends=True)

    return {
        "arg_cast": "".join(arg_cast_lines),
        "body": "".join(body_lines),
    }

def slice_by_line_anchors(lines: list[str], start_pat: str | None, end_pat: str | None) -> tuple[str, list[str]]:
    """
    Returns (chunk, remaining_lines) where chunk is the slice:
      - from first line containing start_pat (inclusive) if start_pat is not None
      - until first line containing end_pat (exclusive) if end_pat is not None
    If start_pat is None => start from beginning.
    If end_pat is None => take everything.
    """
    start_idx = 0
    if start_pat is not None:
        for i, ln in enumerate(lines):
            if start_pat in ln:
                start_idx = i
                break

    end_idx = len(lines)
    if end_pat is not None:
        for j in range(start_idx, len(lines)):
            if end_pat in lines[j]:
                end_idx = j
                break

    chunk = "".join(lines[start_idx:end_idx])
    remaining = lines[end_idx:]
    return chunk, remaining

def parse_pointers_init(block: str) -> List[Tuple[str, str, str]]:
    def split_statements(text: str) -> List[str]:
        # Split by ';' but keep it, and avoid returning empty chunks.
        stmts = []
        buf = []
        for ch in text:
            buf.append(ch)
            if ch == ';':
                stmt = ''.join(buf).strip()
                if stmt:
                    stmts.append(stmt)
                buf = []
        # If trailing junk without ';' exists, ignore or raise; here we ignore.
        return stmts

    out: List[Tuple[str, str, str]] = []

    for stmt in split_statements(block):
        star_i = stmt.find('*')
        eq_i = stmt.find('=', star_i + 1)
        if star_i == -1 or eq_i == -1:
            raise ValueError(f"Not a supported declaration (missing '*' or '='): {stmt!r}")

        type_str = stmt[:star_i + 1].strip()         # include '*'
        var_name = stmt[star_i + 1:eq_i].strip()     # between '*' and '='
        initializer = stmt[eq_i:].strip()            # from '=' to ';' (already included)

        out.append((type_str, var_name, initializer))

    return out

def parse_closure_body_sections(body: str) -> dict[str, str]:
    """
    Parse the big closure body (between '// CLOSURE FUNCTION CALL' and '// CLOSURE ARG WRITEBACK')

    Heuristics/anchors used:
      - measurements init: starts at first 'const static char' and ends before '// Initialize DMA futures'
      - DMA init:          between '// Initialize DMA futures' and '// TILING LOOP'
      - loop:              starts at '// TILING LOOP' and ends at '// CLOSE TILING LOOP'
      - profiling:         between 'StopTimer();' and 'StartTimer();'
    Inside the loop:
      - transfer input:    between '// Transfer input tiles' and '// Wait for input tiles'
      - ingress end:       line with '_ingress_dma_wait_end_measurements' (getCycles end)
      - kernel start/end:  lines with '_kernel_start_measurements' and '_kernel_end_measurements'
      - transfer output:   between '// Transfer output tiles' and '// Wait for output tiles'
      - egress end:        line with '_egress_dma_wait_end_measurements'
      - cluster fork args: first compound literal for '__..._cluster_fork_args_t ... = (...){ ... };'
      - pi_cl_team_fork:   line containing 'pi_cl_team_fork('
    """
    out = {
        "buffers_init": "",
        "pointers_init": "",
        "measurements_init": "",
        "dma_init": "",
        "for_loop_statement": "",
        "dma_in_getCycles_start": "",
        "dma_in_transfer": "",
        "dma_in_getCycles_end": "",
        "kernel_getCycles_start": "",
        "tiling_closure_arg_cast_prefix": "",
        "tiling_closure_arg_cast_args": "",
        "tiling_closure_arg_cast_suffix": "",
        "pi_cl_team_fork_call": "",
        "kernel_getCycles_end": "",
        "dma_out_getCycles_start": "",
        "dma_out_transfer": "",
        "dma_out_getCycles_end": "",
        "end_for_loop_and_ptr_increment": "",
        "profiling_block": "",
    }

    lines = body.splitlines(keepends=True)

    # 1) Split off profiling (StopTimer .. StartTimer)
    profiling = ""
    if "StopTimer();" in body and "StartTimer();" in body:
        pre, rest = body.split("StopTimer();", 1)
        prof, post = rest.split("StartTimer();", 1)
        body_main = pre
        profiling = "StopTimer();" + prof + "StartTimer();"
    else:
        body_main = body
    out["profiling_block"] = profiling

    lines = body_main.splitlines(keepends=True)

    # 2) buffers_init = start .. before first 'const static char'
    #    (this includes the ref pointer inits etc.)
    buf_and_poin_init, rem = slice_by_line_anchors(lines, None, "const static char")
    buf_and_poin_init_lines = buf_and_poin_init.splitlines(keepends=True)
    buf_init, poin_init = slice_by_line_anchors(buf_and_poin_init_lines, None, "void *")
    out["buffers_init"] = buf_init
    out["pointers_init"] = parse_pointers_init("".join(poin_init))
    lines = rem

    # 3) measurements_init = from first const static char .. before '// Initialize DMA futures'
    meas_init, rem = slice_by_line_anchors(lines, "const static char", "// Initialize DMA futures")
    out["measurements_init"] = meas_init
    lines = rem

    # 4) dma_init = from '// Initialize DMA futures' .. before '// TILING LOOP'
    dma_init, rem = slice_by_line_anchors(lines, "// Initialize DMA futures", "// TILING LOOP")
    out["dma_init"] = dma_init
    lines = rem

    # 5) Loop chunk = from '// TILING LOOP' .. include '// CLOSE TILING LOOP' section
    loop_chunk, rem_after_loop = slice_by_line_anchors(lines, "// TILING LOOP", "// CLOSE TILING LOOP")
    # We still need to include the CLOSE TILING LOOP line + the closing brace that follows,
    # so we take until the end of the loop closing brace if present in remainder.
    close_tail = ""
    if rem_after_loop:
        close_tail += rem_after_loop[0]  # likely the CLOSE TILING LOOP comment line
        rem_after_loop = rem_after_loop[1:]

        # include following '}' line if present
        if rem_after_loop and rem_after_loop[0].lstrip().startswith("}"):
            close_tail += rem_after_loop[0]
            rem_after_loop = rem_after_loop[1:]
    loop_full = loop_chunk + close_tail

    # remaining after loop: ptr increment + deinit (often)
    remaining_after_loop = "".join(rem_after_loop)
    out["end_for_loop_and_ptr_increment"] = remaining_after_loop

    # ---- Now parse inside the loop ----
    loop_lines = loop_full.splitlines(keepends=True)

    # for_loop_statement: from '// TILING LOOP' up to opening '{' line (inclusive)
    stmt_lines = []
    i = 0
    while i < len(loop_lines):
        stmt_lines.append(loop_lines[i])
        if "{" in loop_lines[i]:
            i += 1
            break
        i += 1
    out["for_loop_statement"] = "".join(stmt_lines)

    loop_rest = loop_lines[i:]

    # dma_in_getCycles_start: take first assignment to _ingress_dma_wait_start_measurements[...] = getCycles();
    start_lines = []
    j = 0
    while j < len(loop_rest):
        start_lines.append(loop_rest[j])
        if "_ingress_dma_wait_start_measurements" in loop_rest[j] and "getCycles" in loop_rest[j]:
            j += 1
            break
        j += 1
    out["dma_in_getCycles_start"] = "".join(start_lines)
    loop_rest = loop_rest[j:]

    din_lines = []
    k = 0
    started = False

    while k < len(loop_rest):
        line = loop_rest[k]

        if not started:
            if "// Transfer input tiles" in line:
                started = True
                din_lines.append(line)
            k += 1
            continue

        # stop RIGHT BEFORE the measurement write
        if "_ingress_dma_wait_end_measurements" in line and "getCycles" in line:
            break

        din_lines.append(line)
        k += 1

    out["dma_in_transfer"] = "".join(din_lines)

    # dma_in_getCycles_end: exactly the measurement write
    if k < len(loop_rest):
        out["dma_in_getCycles_end"] = loop_rest[k]
        k += 1
    else:
        out["dma_in_getCycles_end"] = ""

    loop_rest = loop_rest[k:]

    # kernel_getCycles_start: capture until kernel_start_measurements[...] = getCycles();
    ks_lines = []
    k = 0
    while k < len(loop_rest):
        ks_lines.append(loop_rest[k])
        if "_kernel_start_measurements" in loop_rest[k] and "getCycles" in loop_rest[k]:
            k += 1
            break
        k += 1
    out["kernel_getCycles_start"] = "".join(ks_lines)
    loop_rest = loop_rest[k:]

    # tiling_closure_arg casting: find first cluster_fork args compound literal + split it
    rest_str = "".join(loop_rest)
    pattern_cluster_fork_args = re.compile(
        r"\b\w+_cluster_fork_args_t\s+\w+\s*=\s*\(\s*\w+_cluster_fork_args_t\s*\)\s*\{",
        re.MULTILINE
    )
    m = re.search(
        pattern_cluster_fork_args,
        rest_str
    )

    if m:
        start = m.start()

        # Everything before the compound literal: KEEP IT as its own chunk
        before = rest_str[:start]
        out["pre_cluster_fork_setup"] = before

        # Now parse the compound literal itself
        tail = rest_str[start:]
        end_pos = tail.find("};")
        if end_pos != -1:
            compound = tail[: end_pos + 2]   # include "};"
            after = tail[end_pos + 2 :]

            prefix, args, suffix = split_compound_literal(compound)

            out["tiling_closure_arg_cast_prefix"] = prefix
            out["tiling_closure_arg_cast_args"]   = args
            out["tiling_closure_arg_cast_suffix"] = suffix

            # Continue parsing AFTER the compound literal only
            loop_rest = after.splitlines(keepends=True)
        else:
            # If malformed, keep remaining content so you can debug
            loop_rest = tail.splitlines(keepends=True)
    else:
        # No cluster_fork compound found: keep loop_rest unchanged
        out["pre_cluster_fork_setup"] = ""

    # pi_cl_team_fork call: capture the line containing it (and maybe surrounding whitespace)
    fork_lines = []
    new_rest = []
    taken_fork = False
    for ln in loop_rest:
        if (not taken_fork) and "pi_cl_team_fork" in ln:
            fork_lines.append(ln)
            taken_fork = True
        else:
            new_rest.append(ln)
    out["pi_cl_team_fork_call"] = "".join(fork_lines)
    loop_rest = new_rest

    # kernel_getCycles_end: capture until kernel_end_measurements[...] = getCycles();
    ke_lines = []
    k = 0
    while k < len(loop_rest):
        ke_lines.append(loop_rest[k])
        if "_kernel_end_measurements" in loop_rest[k] and "getCycles" in loop_rest[k]:
            k += 1
            break
        k += 1
    out["kernel_getCycles_end"] = "".join(ke_lines)
    loop_rest = loop_rest[k:]

    # dma_out_getCycles_start: capture until egress_dma_wait_start_measurements[...] = getCycles();
    eos_lines = []
    k = 0
    while k < len(loop_rest):
        eos_lines.append(loop_rest[k])
        if "_egress_dma_wait_start_measurements" in loop_rest[k] and "getCycles" in loop_rest[k]:
            k += 1
            break
        k += 1
    out["dma_out_getCycles_start"] = "".join(eos_lines)
    loop_rest = loop_rest[k:]

    # dma_out_transfer: between '// Transfer output tiles' and '// Wait for output tiles'
    dout, rem3 = slice_by_line_anchors(loop_rest, "// Transfer output tiles", "// Wait for output tiles")
    out["dma_out_transfer"] = dout
    loop_rest = rem3

    # dma_out_getCycles_end: capture until egress_dma_wait_end_measurements[...] = getCycles();
    eoe_lines = []
    k = 0
    while k < len(loop_rest):
        if "_egress_dma_wait_end_measurements" in loop_rest[k] and "getCycles" in loop_rest[k]:
            break
        eoe_lines.append(loop_rest[k])
        k += 1
    out["dma_out_getCycles_end"] = "".join(eoe_lines)
    # print(f"out[dma_out_getCycles_end] = \n{out['dma_out_getCycles_end']}\n")

    return out

def remove_increment_statements(code: str) -> str:

    lines = code.splitlines(keepends=True)
    out = []

    dropping = False         # currently dropping a += statement until ';'
    pending_lhs = False      # saw something that could be LHS and next lines may contain '+='

    def is_brace_only(s: str) -> bool:
        t = s.strip()
        return t in ("{", "}", "};")

    for line in lines:
        stripped = line.strip()

        # Always keep pure braces
        if not dropping and is_brace_only(line):
            out.append(line)
            pending_lhs = False
            continue

        if dropping:
            # keep dropping until we hit a semicolon ending the statement
            if ";" in line:
                # end dropping after the first ';' (rest of line after ';' is rare in codegen)
                # If you want to preserve trailing text after ';', we can, but usually unnecessary.
                dropping = False
            continue

        # Not dropping: detect '+=' on this line
        if "+=" in line:
            dropping = True
            # if the statement ends on same line, stop dropping immediately
            if ";" in line:
                dropping = False
            pending_lhs = False
            continue

        # Detect a standalone line that is only '+=' (split operator case)
        # e.g. line.strip() == "+=" or " +=" etc.
        if stripped == "+=":
            dropping = True
            pending_lhs = False
            continue

        # Heuristic: if previous line looked like a LHS continuation, and this line contains '+=' possibly spaced
        # We'll just rely on the "+=" / stripped == "+=" checks above; they cover the split operator.
        # pending_lhs is kept if you want to extend this later.

        out.append(line)

    return "".join(out)

def split_tiling_call_body(call_body: str) -> dict[str, str]:
    """
    Split tiling_closure call_body (self.tiling_closure[1][1]) into:
      - arg_cast: from beginning to just before '// CLOSURE FUNCTION CALL'
      - function_body: from after '// CLOSURE FUNCTION CALL' to end
                       (or stop before '// CLOSURE ARG WRITEBACK' if present)
    """

    FUNC_CALL_MARKER = "// CLOSURE FUNCTION CALL"
    WRITEBACK_MARKER = "// CLOSURE ARG WRITEBACK"

    lines = call_body.splitlines(keepends=True)

    arg_cast_lines: list[str] = []
    body_lines: list[str] = []

    mode = "arg_cast"
    saw_func_call = False

    for line in lines:
        if mode == "arg_cast":
            if FUNC_CALL_MARKER in line:
                mode = "body"
                saw_func_call = True
                continue  # do not include the marker line itself
            arg_cast_lines.append(line)
        else:
            # mode == "body"
            if WRITEBACK_MARKER in line:
                break
            body_lines.append(line)

    if not saw_func_call:
        # If marker missing, fallback: everything is body (or raise, up to you)
        return {"arg_cast": "", "function_body": call_body}
    
    arg_cast_lines = remove_first_lines("".join(arg_cast_lines)).splitlines(keepends=True)
    body_lines = remove_increment_statements("".join(body_lines)).splitlines(keepends=True)

    return {
        "arg_cast": "".join(arg_cast_lines),
        "function_body": "".join(body_lines),
    }

def replace_num_cores(code: str, nb_cores: int) -> str:
    return code.replace(
        "NUM_CORES",
        f"{nb_cores}"
    )

def normalize_args_cast(arg_str: str) -> list[str]:
    """
    Turns a multi-line args_cast string into a list of
    clean '.field = value' lines with no leading indentation
    and no trailing commas.
    """
    out = []
    for line in arg_str.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(","):
            s = s[:-1]
        out.append(s)
    return out

def compute_tiling_i_ranges(scheduling_queue) -> List[Dict[str, Dict[str, Tuple[int, int]]]]:
    """
    Compute tiling index ranges per round, per model, per layer.

    Returns:
        ranges[round_idx][model_name][layer_name] = (start_tiling_i, end_tiling_i)
    where the interval is half-open: start inclusive, end exclusive.

    Assumes each scheduled item `tile` has attributes:
        - tile.model_name
        - tile.layer_name
    """

    cum = defaultdict(lambda: defaultdict(int))

    ranges_per_round: List[Dict[str, Dict[str, Tuple[int, int]]]] = []

    for round_layers in scheduling_queue:
        round_counts = defaultdict(lambda: defaultdict(int))
        for model_tiles in round_layers:
            for tile in model_tiles:
                round_counts[tile.model_name][tile.layer_name] += 1

        round_ranges: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for model_name, layer_counts in round_counts.items():
            round_ranges[model_name] = {}
            for layer_name, cnt in layer_counts.items():
                start = cum[model_name][layer_name]
                end = start + cnt
                round_ranges[model_name][layer_name] = (start, end)

        for model_name, layer_counts in round_counts.items():
            for layer_name, cnt in layer_counts.items():
                cum[model_name][layer_name] += cnt

        ranges_per_round.append(round_ranges)

    return ranges_per_round

def generate_subgroup_barriers_code():
    return """
#include "hal/eu/eu_v3.h"
#include "archi/eu/eu_v3.h"

static inline uint32_t subgroup_core_mask(uint32_t group_start, uint32_t group_end)
{
    uint32_t mask = 0;

    for (uint32_t c = group_start; c <= group_end; c++) {
        for (volatile int i = 0; i < 8; i++) {
            asm volatile("nop");
        }
        // printf("Adding core %u to subgroup barrier mask (bit %08x)\\n", c, bit);
        mask |= (1u << c);
    }

    return mask;
}

static inline void subgroup_barrier_init(uint32_t barrier_id,
                                         uint32_t group_start,
                                         uint32_t group_end)
{
    uint32_t mask = subgroup_core_mask(group_start, group_end);
    eu_bar_setup(eu_bar_addr(barrier_id), mask);
}

static inline void subgroup_barrier_wait(uint32_t barrier_id, uint32_t core_id)
{
    // printf("Core %u waiting on barrier %u\\n", core_id, barrier_id);
    eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);
    eu_bar_trig_wait_clr(eu_bar_addr(barrier_id));
    // printf("Core %u passed barrier %u\\n", core_id, barrier_id);
}
"""

def generate_timings_printing_code(tiles_timings):
    ret_str = "void print_tiles_timings() {\n"
    for round_idx, round_timings in enumerate(tiles_timings):
        ret_str += f""" printf("=== Round {round_idx} ===\\n");\n"""
        for tile_idx, tile in enumerate(round_timings):
            ret_str += f""" printf("Tile {tile['model_name']}.{tile['layer_name']}\\n[DMA_in] %d cycles\\n[Kernel] %d cycles\\n[DMA_out] %d cycles\\n", round_{round_idx}_tiles_timings[{tile_idx}][0], round_{round_idx}_tiles_timings[{tile_idx}][1], round_{round_idx}_tiles_timings[{tile_idx}][2]);\n"""
        ret_str += f""" printf("\\n");\n"""
    ret_str += "}\n"
    return ret_str