import re
from typing import Dict, Tuple, Any, List

def normalize_layer_name(name: str) -> str:
    """
    Normalize layer names coming from:
      - ModelProfile (suffix _L2, _L3, etc.)
      - FunctionCall (contains closure, memory level)
    """
    if "___" in name:
        parts = name.split("___", 1)
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

DOUBLE_BUFFERING_MARKER = "// DOUBLE BUFFERING CHECK TILE LOAD"


def is_double_buffered_closure(body: str) -> bool:
    """
    Detects whether a layer's closure body was generated with Deeploy's
    `--doublebuffer` tiling (ping-pong prefetch) rather than the default
    single-buffered (SB) shape. Confirmed (by generating and reading real
    output for both a multi-tile and a degenerate single-tile layer) that
    Deeploy always emits this exact marker once per double-buffered input
    tensor, even when the guard it's attached to (`if ((TILING_I+1) <
    numTiles)`) is always false -- so this check is reliable per-layer,
    independent of that layer's actual tile count.
    """
    return DOUBLE_BUFFERING_MARKER in body


def parse_closure_body_sections_db(body: str) -> dict[str, str]:
    """
    Double-buffered (DB) counterpart of `parse_closure_body_sections`. Deeploy's
    `--doublebuffer` tiling emits a structurally different closure body than SB:
    priming DMA transfers for tile 0 BEFORE the loop, a ping-pong buffer-select
    `switch` at the top of every iteration, one "DOUBLE BUFFERING CHECK TILE
    LOAD" block per double-buffered input tensor (prefetching tile i+1 while
    tile i computes), and DMA-out staggered by one iteration (waits for tile
    i-1's output, transfers tile i's) with a final post-loop flush for the
    last tile's output. Confirmed via direct generation+reading of real DB
    output (`AnomalyDetection_heavy` at 8 cores/65536 L1) that the KERNEL
    section (cluster_fork_args struct + pi_cl_team_fork call) and the actual
    compute kernel body (`tiling_closure_body`) are byte-identical to SB's --
    buffering scheme only changes the DMA/loop-prologue shape, never the
    compute itself. So, like the SB parser, this drops the kernel section
    entirely (the redesign inlines `tiling_closure_body` directly instead of
    reusing the per-tile `pi_cl_team_fork` call) and, going further than SB's
    parser, also drops all granular per-tile profiling (getCycles measurement
    writes, the printf loop) -- none of it is reused by the redesigned
    combined closure (SB's version already discards most of it too; DB's
    profiling bookkeeping is materially more involved -- per-tensor prefetch
    timestamps -- for no benefit here, since nothing downstream reads it).

    Returned keys:
      - buffers_init, pointers_init: identical extraction/shape to SB's parser
        (arena-offset pointer declarations + buffer_0/buffer_1/_next pointer
        resets) -- these fully describe DB's ping-pong buffer addresses too,
        since buffer_0/buffer_1 are just pointers_init entries like any other.
      - priming_dma: the pre-loop channel declarations + tile-0 priming
        transfers (real hardware DMA calls -- must be leader-gated).
      - for_loop_statement: the loop header, reused verbatim (same as SB).
      - current_buffer_switch: the top-of-iteration ping-pong `switch`
        selecting tile TILING_I's own buffers -- pure pointer arithmetic, safe
        to run redundantly on every core (needed by the inlined kernel body).
      - dma_in_wait_and_prefetch: "Wait for current input tile" plus every
        per-tensor "DOUBLE BUFFERING CHECK TILE LOAD" block (next-tile
        prefetch) -- all real hardware DMA calls, must be leader-gated as one
        atomic unit (the `_next` pointers each block computes are consumed
        only by that same block's own transfer call, so no finer split is
        needed or safe to assume).
      - dma_out_wait_and_transfer: "Wait for previous output tile" + "Transfer
        current output tile" -- leader-gated, same reasoning.
      - final_output_flush: the post-loop "Wait for final output tile" block --
        flushes the LAST tile's output DMA (never waited on inside the loop,
        since DMA-out is staggered by one iteration) -- leader-gated, and the
        caller must barrier after it before any subsequent layer reuses this
        layer's arena region.
    """
    out = {
        "buffers_init": "",
        "pointers_init": [],
        "priming_dma": "",
        "for_loop_statement": "",
        "current_buffer_switch": "",
        "dma_in_wait_and_prefetch": "",
        "dma_out_wait_and_transfer": "",
        "final_output_flush": "",
        "num_tiles_array_name": "",
    }

    # 1) Strip profiling (StopTimer .. StartTimer) -- discarded, not reused.
    if "StopTimer();" in body and "StartTimer();" in body:
        pre, rest = body.split("StopTimer();", 1)
        _, _post = rest.split("StartTimer();", 1)
        body_main = pre
    else:
        body_main = body

    lines = body_main.splitlines(keepends=True)

    # 2) buffers_init / pointers_init: pointer declarations, up to (not
    # including) "// Initialize DMA future" -- unlike SB, where "const static
    # char" (measurements_init) comes right after the pointers, DB emits
    # priming DMA first (see step 3), so that's the correct end anchor here.
    buf_and_poin_init, rem = slice_by_line_anchors(lines, None, "// Initialize DMA future")
    buf_and_poin_init_lines = buf_and_poin_init.splitlines(keepends=True)
    buf_init, poin_init = slice_by_line_anchors(buf_and_poin_init_lines, None, "void *")
    out["buffers_init"] = buf_init
    out["pointers_init"] = parse_pointers_init("".join(poin_init))
    lines = rem

    # 3) priming_dma: "// Initialize DMA future" .. "const static char".
    # Unlike SB (measurements_init declared BEFORE its "// Initialize DMA
    # futures" section, which holds nothing but channel decls), DB emits the
    # channel decls + tile-0 priming transfers FIRST, with the measurement
    # array declarations coming after -- confirmed directly against real
    # generated output (including a case with tensors too large for one DMA
    # burst, whose priming has extra chunked "// UPDATE VARIABLE ..." + a
    # second `mchan_transfer_1d` per tensor -- harmless here, just more plain
    # lines the anchors below don't need to know about).
    priming, rem = slice_by_line_anchors(lines, "// Initialize DMA future", "const static char")
    out["priming_dma"] = "".join(l for l in priming.splitlines(keepends=True) if "getCycles" not in l)
    lines = rem

    # 4) Skip measurements_init (the per-tile timing array declarations, plus
    # tile 0's own ingress-wait-start timestamp write) -- discarded entirely,
    # nothing downstream reuses them.
    _, rem = slice_by_line_anchors(lines, "const static char", "// TILING LOOP")
    lines = rem

    # 5) Loop chunk: "// TILING LOOP" .. "// CLOSE TILING LOOP" (+ trailing
    # close brace) -- same tail-capture logic as SB's parser.
    loop_chunk, rem_after_loop = slice_by_line_anchors(lines, "// TILING LOOP", "// CLOSE TILING LOOP")
    close_tail = ""
    if rem_after_loop:
        close_tail += rem_after_loop[0]
        rem_after_loop = rem_after_loop[1:]
        if rem_after_loop and rem_after_loop[0].lstrip().startswith("}"):
            close_tail += rem_after_loop[0]
            rem_after_loop = rem_after_loop[1:]
    loop_full = loop_chunk + close_tail

    # Everything after the loop: "*tileIdxPtr += 1;", "// Wait for final
    # output tile" (the real flush -- kept), "// Deinitialize DMA future",
    # a trailing measurement, then the printf profiling loop (all discarded).
    final_lines = "".join(rem_after_loop).splitlines(keepends=True)
    flush_chunk, _ = slice_by_line_anchors(final_lines, "// Wait for final output tile", "// Deinitialize DMA future")
    out["final_output_flush"] = flush_chunk

    loop_lines = loop_full.splitlines(keepends=True)

    # for_loop_statement: up to and including the opening '{'.
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

    # num_tiles_array_name: the loop bound is a runtime lookup into a
    # per-layer `..._numTiles[]` array (its actual VALUES are declared
    # elsewhere, in the GLOBALS section, not visible from this closure-body
    # text alone) -- capture just the array's name here so a caller with
    # access to the full file text (SourceFile.py) can resolve the literal
    # tile count and detect the degenerate numTiles==1 case, where DB's
    # deferred/staggered output-DMA flush provides no overlap benefit at all
    # and (confirmed by direct GVSoC-level debugging) is where a real,
    # repeated-invocation hang occurs.
    m = re.search(r"(\w+_numTiles)\[", out["for_loop_statement"])
    if m:
        out["num_tiles_array_name"] = m.group(1)

    # current_buffer_switch: up to (not including) the first "// Wait for
    # current input tile" marker -- includes the optional `if (TILING_I > 0)
    # { ...getCycles... }` profiling guard, whose getCycles line is dropped
    # (harmless either way: a local, per-core array write nothing reads), and
    # the ping-pong buffer-select switch itself (kept).
    switch_lines = []
    k = 0
    while k < len(loop_rest) and "// Wait for current input tile" not in loop_rest[k]:
        switch_lines.append(loop_rest[k])
        k += 1
    out["current_buffer_switch"] = "".join(l for l in switch_lines if "getCycles" not in l)
    loop_rest = loop_rest[k:]

    # dma_in_wait_and_prefetch: from the first "// Wait for current input
    # tile" through the last "DOUBLE BUFFERING CHECK TILE LOAD" block, i.e.
    # up to (not including) the ingress-wait-end measurement line.
    din_lines = []
    k = 0
    while k < len(loop_rest):
        line = loop_rest[k]
        if "_ingress_dma_wait_end_measurements" in line and "getCycles" in line:
            break
        din_lines.append(line)
        k += 1
    out["dma_in_wait_and_prefetch"] = "".join(din_lines)
    loop_rest = loop_rest[k:]
    if loop_rest and "_ingress_dma_wait_end_measurements" in loop_rest[0]:
        loop_rest = loop_rest[1:]

    # Skip the kernel section entirely (kernel_start measurement,
    # cluster_fork_args struct, pi_cl_team_fork call, kernel_end measurement)
    # -- discarded, same rationale as SB's parser: the redesign inlines
    # tiling_closure_body directly instead. Anchor on the egress-wait-start
    # measurement line, which opens the DMA-out section next.
    k = 0
    while k < len(loop_rest) and not ("_egress_dma_wait_start_measurements" in loop_rest[k] and "getCycles" in loop_rest[k]):
        k += 1
    loop_rest = loop_rest[k:]
    if loop_rest and "_egress_dma_wait_start_measurements" in loop_rest[0]:
        loop_rest = loop_rest[1:]

    # dma_out_wait_and_transfer: "// Wait for previous output tile" + "//
    # Transfer current output tile", up to (not including) the egress-wait-end
    # measurement line.
    dout_lines = []
    k = 0
    while k < len(loop_rest):
        line = loop_rest[k]
        if "_egress_dma_wait_end_measurements" in line and "getCycles" in line:
            break
        dout_lines.append(line)
        k += 1
    out["dma_out_wait_and_transfer"] = "".join(dout_lines)

    return out


def lookup_num_tiles(full_file_content: str, num_tiles_array_name: str) -> "int | None":
    """
    Resolves a DB layer's literal tile count from its `..._numTiles[]` array
    DECLARATION (in the GLOBALS section, e.g. `static PI_L1 uint8_t
    ..._numTiles[2] = {0, 4};` -- L2-only/single-L3-chunk tiling always has
    exactly 2 entries, {0, N}, since tileIdxPtr only ever takes the values 0
    and 1 in that case). `parse_closure_body_sections_db` only sees the
    closure body text, not the globals section where this is declared, so
    this needs the FULL source file text to resolve -- returns None if the
    array name is empty (SB layers don't have one) or the declaration isn't
    found (defensive; callers should treat that as "unknown, assume multi-tile"
    rather than crash).
    """
    if not num_tiles_array_name:
        return None
    m = re.search(
        rf"\b{re.escape(num_tiles_array_name)}\[\d+\]\s*=\s*\{{\s*0\s*,\s*(\d+)\s*\}}",
        full_file_content,
    )
    return int(m.group(1)) if m else None


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
    // Deliberately does NOT clear PULP_HW_BAR_EVENT from the mask after use
    // (a fix attempted and reverted -- see git history/session notes): PULP-OS's
    // OWN pi_cl_team_barrier()/pi_cl_team_fork() join-wait (pos_team_offload_wait)
    // call eu_bar_trig_wait_clr() DIRECTLY, without ever arming
    // PULP_HW_BAR_EVENT themselves -- they rely on it staying permanently
    // enabled once armed here. Clearing it after use breaks the very next
    // pi_cl_team_fork's join-wait (used to dispatch every Combined_closure_*),
    // which then hangs forever since nothing re-arms the bit for it. Confirmed
    // by tracing pi_cl_team_barrier()'s implementation
    // (pulp_hal/.../pos/implem/pe.h) after a mask-clear "fix" here caused
    // exactly that hang.
    eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);
    eu_bar_trig_wait_clr(eu_bar_addr(barrier_id));
    // printf("Core %u passed barrier %u\\n", core_id, barrier_id);
}
"""