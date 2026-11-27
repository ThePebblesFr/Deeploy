from typing import Dict, List, Tuple
from MultiDeeploy.utils import replace_num_cores


def _split_statements(text: str) -> List[str]:
    stmts = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch == ';':
            stmt = "".join(buf)
            if stmt.strip():
                stmts.append(stmt)
            buf = []
    if buf and "".join(buf).strip():
        stmts.append("".join(buf))
    return stmts


def split_buffers_init_for_globals(buffers_init_text: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    A tile's `buffers_init` text (from LayerFunctionDefinition.closure_body[1])
    is a blob of "TYPE *NAME = init;"-shaped local pointer declarations,
    copied verbatim from Deeploy's original single-model closure. In the old
    one-giant-function combined closure, these were genuinely local -- but
    since some of them are READ by a LATER layer's own body without that
    layer redeclaring them itself (a real cross-layer handoff relying on them
    still being in scope, confirmed via a real "use of undeclared identifier"
    compile failure once layers became separate functions -- see
    layer_function_definition_statement's docstring for why they must be
    separate functions at all), every declaration found here must be promoted
    to a file-scope global instead of staying local to whichever layer's
    function happens to declare it first.

    Returns (global_decls, local_text):
      - global_decls: (type_str, var_name) pairs to declare once, at file
        scope, before any layer function (deduped across all tiles by the
        caller).
      - local_text: the SAME statements, minus their type prefix (so
        `NAME = init;`, an assignment into the now-global variable, instead
        of a fresh declaration) -- what actually gets emitted inside the
        owning tile's own function body, in original order. A statement that
        doesn't match the "TYPE *NAME = init;" shape (no unambiguous
        '*'-then-'=' split, or the extracted name isn't a plain identifier)
        is assumed to be genuinely tile-local scratch and is passed through
        unchanged.
    """
    global_decls: List[Tuple[str, str]] = []
    local_lines: List[str] = []

    for stmt in _split_statements(buffers_init_text):
        core = stmt.strip()
        if not core:
            local_lines.append(stmt)
            continue
        star_i = core.find('*')
        eq_i = core.find('=', star_i + 1)
        if star_i == -1 or eq_i == -1:
            local_lines.append(stmt)
            continue
        type_str = core[:star_i + 1].strip()
        var_name = core[star_i + 1:eq_i].strip()
        initializer = core[eq_i:].strip()
        if not var_name.isidentifier():
            local_lines.append(stmt)
            continue
        global_decls.append((type_str, var_name))
        local_lines.append(f"  {var_name} {initializer}\n")

    return global_decls, "".join(local_lines)


def dedup_lines_preserve_order(lines):
    """
    Dedup by logical (';'-terminated) statement, not by raw physical line.
    Long, deeply-nested identifiers (ResNet's "_MERGE_CONVRQ_PASS_N_..." names,
    never seen with the short MLP-only models -- AnomalyDetection/TinyClassifier
    -- this pipeline used everywhere else) get wrapped by clang-format such
    that a declaration's TYPE alone can end up alone on its own line, e.g.:
        uint8_t *
            SomeVeryLongName_tileIdxPtr =
                args->SomeVeryLongName_tileIdxPtr;
    A bare line like "uint8_t *" is generic, content-free text that recurs
    identically across many genuinely DIFFERENT declarations. Deduping at the
    physical-line level (the previous implementation) treats those recurring
    fragments as duplicates of each other and drops all but the first,
    silently orphaning every later declaration's remaining lines with no type
    prefix at all -- a real, silent codegen-corruption bug (confirmed via a
    "use of undeclared identifier" compile failure once ResNet was plugged
    in). Splitting into logical statements first makes the dedup key the
    WHOLE (whitespace-normalized) declaration, never just one generic-looking
    fragment of it.
    """
    if isinstance(lines, str):
        split = True
        text = lines
    else:
        split = False
        text = "".join(lines)

    statements = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch == ';':
            statements.append("".join(buf))
            buf = []
    if buf:
        statements.append("".join(buf))

    seen = set()
    out = []
    for stmt in statements:
        key = " ".join(stmt.split())
        if key == "":
            # Whitespace-only trailing chunk (e.g. blank lines before the
            # closing brace) -- always keep, never dedup.
            out.append(stmt)
            continue
        if key not in seen:
            seen.add(key)
            out.append(stmt)

    result = "".join(out)
    return result if split else result.splitlines(keepends=True)


def combined_closure_struct_statement(all_tiles, cores0, cores1):
    """
    One args struct for the whole combined closure, covering every layer of both
    models -- struct fields are just concatenated/deduped text regardless of which
    model a layer belongs to, so this flattens across both instead of building a
    separate struct per model/round the way the old round-based design did.

    Uses ONLY `closure[0][1]` (each layer's original struct fields, i.e. what
    RunNetwork() itself passes in -- inter-layer tensor handoffs and tileIdxPtr).
    Deliberately excludes `tiling_closure[0][1]`: in Deeploy's own generated code,
    those fields exist solely to pass locally-computed pointers from the old
    "closure" function to the nested "cluster_fork"/"tiling_closure" functions it
    forked into -- an internal hop this design collapses away (buffers_init now
    runs directly inside this same combined closure, per layer, so there's no
    longer a boundary for those fields to cross).
    """
    ret_str = "typedef struct {\n"
    temp_str = ""
    for tile in all_tiles:
        temp_str += f"{tile.closure[0][1]}\n"
    ret_str += dedup_lines_preserve_order(temp_str)
    ret_str += f"}} Combined_closure_{cores0}_{cores1}_args_t;\n"
    return ret_str


def generate_channels_global_declaration(model_name):
    """
    File-scope counterpart of the old function-local channel_input/output
    declaration (see combined_closure_function_definition_statement's
    docstring): once each model's layers become separate, non-inlined
    functions (to keep clang's -O3 per-function cost tractable on real CNNs --
    a single ~2800-line combined-closure function was observed to blow clang's
    optimizer up to 35+ GB RSS / 50+ minutes for a KS+VWW pair, vs. negligible
    cost for the tiny MLP pairs this design was first proven on), this handle
    must persist across those function-call boundaries instead of living on
    one function's stack. Only the leader core of this model's group ever
    reads/writes it (see _layer_body_statement's leader_check gating), so a
    plain (non-atomic) global is safe: every other core's copy was already
    dead/unused even when this was a per-core local. Declared with no
    initializer (reset explicitly every RunNetwork() call instead -- see
    generate_channels_reset_statement) since a file-scope initializer would
    only run once at program load, not on every inference.
    """
    return f"uint32_t {model_name}_channel_input;\nuint32_t {model_name}_channel_output;\n"


def generate_channels_reset_statement(model_name):
    """
    Replaces the old local declaration's own `= (uint32_t)-1` initializer,
    now that the variable is a persistent global (see
    generate_channels_global_declaration) -- run once per RunNetwork() call,
    by every core in the model's group (redundant, harmless: same value,
    and only the leader ever actually reads it back).
    """
    return f"    {model_name}_channel_input = (uint32_t)-1;\n    {model_name}_channel_output = (uint32_t)-1;\n"


def _layer_body_statement(tile_def, model_dedicated_cores: int, barrier_id: int, leader_check: str) -> str:
    """
    Emits one layer's complete, un-split tile loop (all of its tiles, in one
    pass -- no more per-round TILING_I range splitting, since each layer now
    always runs its full range in a single place). Reuses SourceFile.py's
    `for_loop_statement` fragment (the layer's own original loop header,
    including its own `numTiles[*tileIdxPtr]` bounds) verbatim -- confirmed it
    doesn't reference the per-tile measurement arrays (`_ingress_dma_wait_*`/
    `_kernel_*`/`_egress_dma_wait_*`), so no declarations for those need
    restoring alongside it.

    Does NOT reuse `end_for_loop_and_ptr_increment` for the loop's closing
    brace: despite the name, that fragment does not actually contain the `}`
    that closes the `for` loop (a pre-existing quirk in
    parse_closure_body_sections()'s extraction -- the closing brace is silently
    dropped between the DMA-out and end-of-loop extraction steps). This is why
    the CURRENT round-based `tiling_closure_function_definition_statement` never
    uses either of these two keys at all: it hardcodes its own `for (...)` header
    (with round-specific integer bounds) and its own closing `}`. This code
    does the same for the closing `}`, while still reusing the ORIGINAL
    `for_loop_statement` header (to avoid needing any tile-count computation).
    Also matches that same current code in never incrementing `*tileIdxPtr`:
    for L2-only tiling (single tile "chunk", our only case today), it starts at
    0 and is never read again in a way that would need advancing -- each
    RunNetwork() call is meant to reprocess the SAME tile range every time,
    which is exactly what leaving it unincremented gives. `buffers_init` (arena
    offset computations) and the pointers_init RESET statements (not the
    declarations, which are emitted once globally by SourceFusion.py) run once
    per layer, same as the arena-allocation step the old round-based
    `closure_function_definition_statement` used to do once per round.

    DMA-in/DMA-out stay leader-only (only `leader_check` core issues them) --
    this is NOT a straight reuse of the solo/unfused code (which does DMA on
    every core redundantly, harmless when solo, but a resource-contention risk
    once co-resident with another model's cores also doing DMA) -- it matches
    the already-proven leader-only pattern the round-based fusion already used.
    The kernel body (`tiling_closure_body[1]`) is inlined directly for every
    core in the group, same as today (no nested pi_cl_team_fork -- that per-tile
    fork is what made a shared, non-arbitrary-range `cores_map` necessary in the
    first place; inlining removes the need for it entirely).
    """
    kernel_body = replace_num_cores(tile_def.tiling_closure_body[1], model_dedicated_cores)

    ret_str = tile_def.closure_body[1]["buffers_init"]
    for p in tile_def.closure_body[1]["pointers_init"]:
        ret_str += f"{p[1].strip()}{p[2].strip()}\n"

    ret_str += tile_def.closure_body[1]["for_loop_statement"]
    ret_str += f"        if ({leader_check}) {{\n"
    ret_str += tile_def.closure_body[1]["dma_in_transfer"]
    ret_str += tile_def.closure_body[1]["pre_cluster_fork_setup"]
    ret_str += "        }\n"
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += kernel_body
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += f"        if ({leader_check}) {{\n"
    ret_str += tile_def.closure_body[1]["dma_out_transfer"]
    ret_str += tile_def.closure_body[1]["dma_out_getCycles_end"]
    ret_str += "        }\n"
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += "      }\n"
    return ret_str


def _layer_body_statement_db(tile_def, model_dedicated_cores: int, barrier_id: int, leader_check: str) -> str:
    """
    Double-buffered (DB) counterpart of `_layer_body_statement`, for a layer
    whose closure_body[1] came from `parse_closure_body_sections_db` (i.e.
    `tile_def.buffering_mode == "DB"`, generated via Deeploy's
    `--doublebuffer` tiling). Same overall shape as the SB version --
    buffers_init/pointers_init run on every core, DMA is leader-gated, the
    inlined kernel body runs on every core, barriers separate each phase --
    but with DB's extra structure: a per-iteration ping-pong buffer-select
    switch (pure pointer arithmetic, every core, not leader-gated -- the
    inlined kernel body on every core needs the resulting A_ref/B_ref/...
    pointers, and the switch has no hardware side effects so redundant
    execution is safe, exactly like buffers_init/pointers_init already are),
    a leader-gated priming DMA before the loop (tile 0's prefetch), and a
    leader-gated post-loop flush for the last tile's staggered output DMA
    (never waited on inside the loop body itself) followed by one extra
    barrier before the next layer -- DB's DMA-out is deferred by one
    iteration, so unlike SB, the loop's own closing brace does NOT imply the
    last tile's output DMA is complete yet.

    Does NOT reuse `dma_in_wait_and_prefetch`/`dma_out_wait_and_transfer` as
    two separately-barriered blocks: both are leader-only real hardware DMA
    calls with no cross-core dependency until their *effects* (the buffers
    they fill) are consumed by the barrier-separated kernel/next-iteration
    code, so each is emitted as a single leader-gated unit, matching how SB's
    `dma_in_transfer`+`pre_cluster_fork_setup` and `dma_out_transfer`+
    `dma_out_getCycles_end` are each already treated as one leader-gated unit.
    """
    kernel_body = replace_num_cores(tile_def.tiling_closure_body[1], model_dedicated_cores)
    sections = tile_def.closure_body[1]

    ret_str = sections["buffers_init"]
    for p in sections["pointers_init"]:
        ret_str += f"{p[1].strip()}{p[2].strip()}\n"

    ret_str += f"      if ({leader_check}) {{\n"
    ret_str += sections["priming_dma"]
    ret_str += "      }\n"

    ret_str += sections["for_loop_statement"]
    ret_str += sections["current_buffer_switch"]
    ret_str += f"        if ({leader_check}) {{\n"
    ret_str += sections["dma_in_wait_and_prefetch"]
    ret_str += "        }\n"
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += kernel_body
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += f"        if ({leader_check}) {{\n"
    ret_str += sections["dma_out_wait_and_transfer"]
    ret_str += "        }\n"
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += "      }\n"  # close for loop

    ret_str += f"      if ({leader_check}) {{\n"
    ret_str += sections["final_output_flush"]
    ret_str += "      }\n"
    ret_str += f"      subgroup_barrier_wait({barrier_id}, core_id);\n"
    return ret_str


def _layer_body_statement_db_single_tile(tile_def, model_dedicated_cores: int, barrier_id: int, leader_check: str) -> str:
    """
    Degenerate-tile-count variant of `_layer_body_statement_db`, for a DB
    layer whose `tile_def.num_tiles == 1`. With only one tile there is no
    "next tile" to overlap with -- every `if ((TILING_I+1) < numTiles)` guard
    in `dma_in_wait_and_prefetch` evaluates false at runtime regardless, so
    keeping those blocks is harmless (dead code, not wrong code) and they are
    reused as-is here as with the general DB path.

    The one real difference: DB's staggered/deferred output-DMA flush
    (`final_output_flush`, unconditional, leader-only, emitted AFTER the loop
    closes and gated behind its own extra barrier) was confirmed via direct
    GVSoC-level tracing (phase-tagged printf instrumentation showing ALL 4
    group members completing their only tile but NONE reaching the barrier
    after the flush) to hang the leader core inside its `mchan_channel_wait`
    on repeated (not first) invocations of this exact code -- e.g. across
    many ticks of the DynMultiDeeploy scenario harness, though a single
    inference already exercises this same flush without issue, consistent
    with some slowly-accumulating GVSoC-level hardware-model state rather
    than a logic bug in the generated code itself.

    Since there is only one tile, deferring the flush buys nothing anyway
    (SB's own immediate-wait behavior is already optimal here) -- so this
    variant does what SB does: issues the transfer and waits for it
    immediately, in the SAME leader-gated block, inside the loop, instead of
    deferring to a separate post-loop block. This needs only 3
    subgroup_barrier_wait calls per layer (matching SB), not 4, and
    completely avoids the code path where the hang was pinpointed.
    """
    kernel_body = replace_num_cores(tile_def.tiling_closure_body[1], model_dedicated_cores)
    sections = tile_def.closure_body[1]

    ret_str = sections["buffers_init"]
    for p in sections["pointers_init"]:
        ret_str += f"{p[1].strip()}{p[2].strip()}\n"

    ret_str += f"      if ({leader_check}) {{\n"
    ret_str += sections["priming_dma"]
    ret_str += "      }\n"

    ret_str += sections["for_loop_statement"]
    ret_str += sections["current_buffer_switch"]
    ret_str += f"        if ({leader_check}) {{\n"
    ret_str += sections["dma_in_wait_and_prefetch"]
    ret_str += "        }\n"
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += kernel_body
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += f"        if ({leader_check}) {{\n"
    ret_str += sections["dma_out_wait_and_transfer"]
    ret_str += sections["final_output_flush"]
    ret_str += "        }\n"
    ret_str += f"        subgroup_barrier_wait({barrier_id}, core_id);\n"
    ret_str += "      }\n"  # close for loop
    return ret_str


def layer_function_definition_statement(
    tile_def, model_dedicated_cores: int, barrier_id: int, leader_check: str, func_name: str, args_type_name: str
) -> str:
    """
    Wraps one layer's body (see _layer_body_statement / _layer_body_statement_db
    / _layer_body_statement_db_single_tile) in its own real, non-inlined
    function, instead of pasting it directly into one shared function body
    alongside every other layer of both models.

    Why this exists: combined_closure_function_definition_statement used to
    inline every layer of both models into ONE function (a real CNN pair, e.g.
    KeywordSpotting_small x VisualWakeWords_small, produces a ~2800-line single
    function this way). clang -O3's per-function optimizer cost does not scale
    linearly with function size -- confirmed empirically: that single function
    drove one clang invocation to 35+ GB RSS and 50+ minutes of CPU time before
    being killed, for a pair whose *combined* source file is only ~330 KB.
    Splitting into one real function per layer keeps every individual
    optimizer unit small (a single layer's tile loop), which is what actually
    fixes the blowup -- merely reformatting the SAME one giant function
    wouldn't. `__attribute__((noinline))` is required, not decorative: clang's
    inliner will happily re-inline a same-translation-unit, called-once static
    function back into its single caller at -O3, silently recreating the exact
    same giant-function blowup this split is meant to avoid.

    A layer's body is not fully self-contained: LayerFunctionDefinition.
    closure_body[0] (the layer's own original single-model arg-cast section)
    and closure_body[1]["buffers_init"] both contain real "TYPE *NAME =
    args->NAME;"-shaped declarations, and some of those NAMEs are read by a
    LATER layer's body without that later layer redeclaring them itself --
    real cross-layer handoff that relied on every layer sharing one C block
    scope in the old single-function design (confirmed via a real "use of
    undeclared identifier" compile failure once layers became separate
    functions). combined_closure_function_definition_statement pre-processes
    both fields (via split_buffers_init_for_globals) before this function
    ever runs, promoting every such declaration to a file-scope global and
    leaving only the plain assignment (`NAME = args->NAME;`, no type) behind
    in the field -- so by the time this function reads closure_body[0]/
    closure_body[1]["buffers_init"], they are already safe to emit as
    ordinary statements inside this one function. `args` and `core_id`
    themselves are passed as plain parameters, no promotion needed for
    either. The one piece of state that DOES need to outlive a single layer's
    function call -- each model's DMA channel handle, reused across that
    model's own successive layers -- is handled separately, as a file-scope
    global (see generate_channels_global_declaration).
    """
    if getattr(tile_def, "buffering_mode", "SB") == "DB":
        if getattr(tile_def, "num_tiles", None) == 1:
            body = _layer_body_statement_db_single_tile(tile_def, model_dedicated_cores, barrier_id, leader_check)
        else:
            body = _layer_body_statement_db(tile_def, model_dedicated_cores, barrier_id, leader_check)
    else:
        body = _layer_body_statement(tile_def, model_dedicated_cores, barrier_id, leader_check)

    ret_str = f"static void __attribute__((noinline)) {func_name}({args_type_name}* args, int core_id) {{\n"
    ret_str += tile_def.closure_body[0]
    ret_str += body
    ret_str += "}\n"
    return ret_str


def combined_closure_function_definition_statement(
    tiles_by_model: List[List], model_names: List[str], models_dedicated_cores: List[int]
) -> str:
    """
    Emits one small dispatcher function per config (Combined_closure_{cores}),
    preceded by one real, non-inlined function per layer of both models (see
    layer_function_definition_statement's docstring for why layers are no
    longer inlined directly into the dispatcher). Each model's cores still run
    their own complete layer sequence, in original order, guarded by a literal
    core-range check (baked in at generation time, not read from a shared
    runtime array) and synced only within their own subgroup barrier -- never
    interleaved tile-by-tile with the other model the way the old round-based
    version was. See SourceFusion.py's module-level notes for the rest of the
    rationale (removes N_rounds-1 extra pi_cl_team_fork join points, and fixes
    a real, pre-existing bug where DynMultiDeeploy's second-level fusion
    silently reused ONE hardcoded core split, from a shared `cores_map`
    global, for every one of the 4 modes -- eliminated here since core ranges
    are now literal constants baked per-config into this function's own text,
    with no shared array to go stale or collide across configs).

    Args:
        tiles_by_model: one list per model, of that model's LayerFunctionDefinition
            objects in original RunNetwork() order (no cross-model interleaving).
        model_names: model_name per model, same order (for the per-model DMA
            channel variable declarations and per-layer function naming --
            already unique per (variant, cores) combination across every mode
            of a standard 4-mode config, confirmed by construction: two modes
            sharing a core split always differ in variant, and vice versa).
        models_dedicated_cores: dedicated core count per model, same order.
    """
    cores0, cores1 = models_dedicated_cores[0], models_dedicated_cores[1]
    args_type_name = f"Combined_closure_{cores0}_{cores1}_args_t"

    nb_models = len(tiles_by_model)
    core_cursor = 0
    group_ranges: List[Tuple[int, int]] = []
    for n in models_dedicated_cores:
        group_ranges.append((core_cursor, core_cursor + n - 1))
        core_cursor += n

    # closure_body[0] (the layer's own original arg-cast section -- often but
    # not always just the redundant "args_t* args = ...;" line; several
    # layers' closure_body[0] instead carries real "TYPE *NAME = args->NAME;"
    # declarations of struct fields consumed as locals) and
    # closure_body[1]["buffers_init"]'s "TYPE *NAME = init;" pointer
    # declarations must both become file-scope globals now that layers are
    # separate functions (see split_buffers_init_for_globals's docstring): a
    # later layer can read a name it never itself declares, relying on an
    # earlier layer having set it -- true even across model boundaries, hence
    # deduping across every tile of every model, not just within one model's
    # own layer sequence. Mutates each tile's own fields in place (dropping
    # the type prefix, keeping the assignment) since
    # layer_function_definition_statement reads them directly.
    buffers_init_global_decls: List[Tuple[str, str]] = []
    seen_global_decl_names = set()
    for model_tiles in tiles_by_model:
        for tile in model_tiles:
            decls0, local_text0 = split_buffers_init_for_globals(tile.closure_body[0])
            tile.closure_body[0] = local_text0
            decls1, local_text1 = split_buffers_init_for_globals(tile.closure_body[1]["buffers_init"])
            tile.closure_body[1]["buffers_init"] = local_text1
            for type_str, var_name in decls0 + decls1:
                if var_name not in seen_global_decl_names:
                    seen_global_decl_names.add(var_name)
                    buffers_init_global_decls.append((type_str, var_name))

    # File-scope globals: the buffers_init pointer decls above, plus one
    # channel-handle pair per model that actually has layers, emitted before
    # any function definition.
    globals_str = ""
    for type_str, var_name in buffers_init_global_decls:
        globals_str += f"{type_str}{var_name};\n"
    globals_str += "\n"
    for model_tiles, model_name in zip(tiles_by_model, model_names):
        if not model_tiles:
            continue
        globals_str += generate_channels_global_declaration(model_name)
    globals_str += "\n"

    # One real, non-inlined function per layer, named {model_name}_layer_{i}.
    layer_funcs_str = ""
    layer_func_names_by_model: List[List[str]] = []
    for model_idx, (model_tiles, model_name) in enumerate(zip(tiles_by_model, model_names)):
        start, end = group_ranges[model_idx]
        leader_check = f"core_id == {start}"
        func_names = []
        for i, tile in enumerate(model_tiles):
            func_name = f"{model_name}_layer_{i}"
            func_names.append(func_name)
            layer_funcs_str += layer_function_definition_statement(
                tile, models_dedicated_cores[model_idx], model_idx + 2, leader_check, func_name, args_type_name
            )
            layer_funcs_str += "\n"
        layer_func_names_by_model.append(func_names)

    # The dispatcher itself: barrier setup, then per-model channel reset +
    # guarded sequential calls into that model's own layer functions.
    ret_str = f"static void Combined_closure_{cores0}_{cores1}({args_type_name}* closure_args) {{\n"
    ret_str += f"    {args_type_name}* args = ({args_type_name}*) closure_args;\n"
    ret_str += "    int core_id = pi_core_id();\n\n"

    ret_str += "    if (core_id == 0) {\n"
    for model_idx, (start, end) in enumerate(group_ranges):
        ret_str += f"        subgroup_barrier_init({model_idx + 2}, {start}, {end});\n"
    ret_str += "    }\n"
    ret_str += f"    subgroup_barrier_init({nb_models + 2}, 0, NUM_CORES - 1);\n"
    ret_str += "    pi_cl_team_barrier();\n"
    ret_str += "    eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);\n\n"

    for model_idx, (model_tiles, model_name, (start, end)) in enumerate(
        zip(tiles_by_model, model_names, group_ranges)
    ):
        if not model_tiles:
            continue

        ret_str += generate_channels_reset_statement(model_name)
        ret_str += f"    if ({start} <= core_id && core_id <= {end}) {{\n"
        for func_name in layer_func_names_by_model[model_idx]:
            ret_str += f"        {func_name}(args, core_id);\n"
        ret_str += "    }\n"

    ret_str += f"    subgroup_barrier_wait({nb_models + 2}, core_id);\n"
    ret_str += "}\n"

    return globals_str + layer_funcs_str + ret_str
