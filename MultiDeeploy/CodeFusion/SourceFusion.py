from typing import List
from MultiDeeploy.CodeFusion.SourceFile import SourceFile
from MultiDeeploy.CodeFusion.Templates.FunctionsDefinition_templates import (
    combined_closure_struct_statement,
    combined_closure_function_definition_statement,
)
from MultiDeeploy.utils import generate_subgroup_barriers_code
from MultiDeeploy.CodeFusion.Templates.RunNetwork_templates import combined_arg_cast_statement, combined_closure_call_statement

# Marker preceding the fused per-config closure code, consumed by
# DynMultiDeeploy/SourceFusion.py's split_Network_c() to find the
# globals/functions boundary -- keep these in sync if either changes.
FUSED_FUNCTIONS_MARKER = "// ===== FUSED FUNCTIONS ====="


class SourceFusion():
    """
    Fuses N models onto disjoint core/L1 shares of one cluster.

    Each model's cores run that model's own complete layer sequence
    independently, in original order, synced only within their own subgroup
    barrier -- NOT interleaved tile-by-tile with the other model(s) the way an
    earlier, round-synchronized design (WorkloadScheduler-driven) used to work.
    That design existed because arbitrary-core-subset barriers weren't
    available; `subgroup_barrier_init`/`_wait` (still used here, just
    re-initialized once per RunNetwork() call instead of once per round) make it
    unnecessary. Removing the round structure also fixes a real, independently-
    discovered bug: since core-range boundaries are now literal constants baked
    directly into each config's own generated code (computed once, below, from
    each model's dedicated core count), there is no shared `cores_map` global
    left for a second-level fusion step (DynMultiDeeploy's) to accidentally
    reuse a stale value from a different config.

    Measured before building this: pure load-rebalancing gain (removing round
    quantization alone, keeping everything else) is 0% for the real
    DynMultiDeeploy configs on hand today (one model's tiles already dominate
    every round). This redesign is still worth it for two other reasons: it
    removes N_rounds-1 extra `pi_cl_team_fork` join points (real overhead the
    load-balance calculation doesn't capture), and it fixes the cores_map bug
    above. It does NOT restore/add DMA-compute overlap (double buffering) --
    that needs new parser capability (SourceFile.py's extraction hard-assumes
    the single-buffered comment-marker shape), not a wiring change, and is
    deferred pending real target models that would actually need it.
    """

    def __init__(self, files: List[SourceFile], output_path: str, models_dedicated_cores: List[int]):
        self.files = files
        self.output_path = output_path
        self.models_dedicated_cores = models_dedicated_cores
        self.fused_content: List[str] = []

    def fuse(self) -> None:
        self.globals_fusion()
        self.functions_fusion()
        self.run_network_fusion()
        self.init_network_fusion()

        # THESIS FIX: rebase absolute core ids on each model's first core
        _names = [s.model_name for s in self.files]
        _starts = [sum(self.models_dedicated_cores[:k]) for k in range(len(self.files))]
        _text = rebase_core_ids("".join(self.fused_content), _names, _starts)
        with open(self.output_path, 'w') as f:
            f.write(_text)

    def globals_fusion(self) -> None:
        seen_lines = set()
        cluster_dev_added = False

        for source in self.files:
            self.fused_content.append(f"\n// ===== GLOBALS from {source.model_name} START =====\n\n")
            globals_part = source.globals.content

            for line in globals_part:
                if source.globals.GLOBALS_END_PATTERN in line:
                    if not cluster_dev_added:
                        self.fused_content.append(line)
                        cluster_dev_added = True
                    continue

                stripped = line.lstrip()
                if stripped.startswith(("#include", "#define")):
                    if line not in seen_lines:
                        self.fused_content.append(line)
                        seen_lines.add(line)
                else:
                    self.fused_content.append(line)
            self.fused_content.append(f"\n// ===== GLOBALS from {source.model_name} END =====\n\n")

        self.fused_content.append("\n")

        # No more cores_map/nb_dedicated_cores globals: each config's combined
        # closure bakes its own core-range boundaries in as literal constants
        # (see combined_closure_function_definition_statement), so there is
        # nothing shared left to declare here -- this is also the cores_map bug
        # fix (see class docstring): a stale/wrong shared value can no longer
        # exist for a second-level fusion step to pick up.
        self.fused_content.append("// ===== SUBGROUP BARRIERS START =====\n\n")
        self.fused_content.append(generate_subgroup_barriers_code())
        self.fused_content.append("// ===== SUBGROUP BARRIERS END =====\n\n")

    def functions_fusion(self) -> None:
        """
        Build a map (model_name, layer_name) -> LayerFunctionDefinition by
        aligning each source's logical layer order (from
        src.run_network.functions_calls) with its parsed function layers (from
        src.functions.layers_functions_definition), then emit ONE combined
        closure covering every model's full layer sequence in that order (see
        combined_closure_function_definition_statement's docstring for the
        per-layer body it emits).
        """
        lf_map = {}
        tiles_by_model: List[List] = []
        model_names: List[str] = []

        for src in self.files:
            ordered = []
            seen = set()
            for fc in src.run_network.functions_calls:
                if fc.layer_name and fc.layer_name not in seen:
                    ordered.append(fc.layer_name)
                    seen.add(fc.layer_name)

            layer_defs = src.functions.layers_functions_definition
            if len(ordered) != len(layer_defs):
                raise RuntimeError(
                    f"[{src.model_name}] Mismatch between logical layers in RunNetwork "
                    f"({len(ordered)}) and parsed function layers ({len(layer_defs)}). "
                    f"Did you call getFunctions() before fusion? Or parsing order differs."
                )

            model_tiles = []
            for layer_name, layer_def in zip(ordered, layer_defs):
                key = (src.model_name, layer_name)
                if key in lf_map:
                    raise RuntimeError(f"Duplicate LayerFunctionDefinition for {key}")
                lf_map[key] = layer_def
                model_tiles.append(layer_def)

            tiles_by_model.append(model_tiles)
            model_names.append(src.model_name)

        all_tiles = [tile for model_tiles in tiles_by_model for tile in model_tiles]

        self.fused_content.append(f"\n{FUSED_FUNCTIONS_MARKER}\n\n")

        # Global pointer declarations, deduped once across the whole fused file
        # (no per-round "first seen" tracking needed anymore: with no round
        # splitting, every layer is processed in exactly one place, exactly
        # once). The matching per-layer RESET/assignment statements are emitted
        # inside the combined closure itself, every time (see
        # combined_closure_function_definition_statement).
        emitted_global_ptr_decls = set()
        for tile in all_tiles:
            for p in tile.closure_body[1]["pointers_init"]:
                decl = f"{p[0].rstrip()}{p[1].strip()};\n"
                if decl not in emitted_global_ptr_decls:
                    emitted_global_ptr_decls.add(decl)
                    self.fused_content.append(decl)
        self.fused_content.append("\n")

        cores0, cores1 = self.models_dedicated_cores[0], self.models_dedicated_cores[1]
        self.fused_content.append(combined_closure_struct_statement(all_tiles, cores0, cores1))
        self.fused_content.append("\n")
        self.fused_content.append(
            combined_closure_function_definition_statement(tiles_by_model, model_names, self.models_dedicated_cores)
        )
        self.fused_content.append("\n")

    def init_network_fusion(self) -> None:
        self.fused_content.append("\nvoid InitNetwork() {\n")

        for source in self.files:
            for line in source.init_network.content:
                self.fused_content.append(line)

        self.fused_content.append("}\n")

    def run_network_fusion(self) -> None:
        """
        Fuse RunNetwork(): variable declarations from every source, then ONE
        arg-cast + ONE dispatch of the combined closure built by
        functions_fusion() (no per-round loop -- see that method and
        combined_closure_function_definition_statement's docstrings).
        """
        self.fused_content.append("\nvoid RunNetwork() {\n")

        for src in self.files:
            for var_stmt in src.run_network.variables:
                self.fused_content.append(var_stmt)
            self.fused_content.append("\n")
        self.fused_content.append("\n")

        # Walk in the same (model, layer) order functions_fusion() used
        # (self.files order, each source's functions_calls already in that
        # model's original layer order).
        list_of_args = []
        list_of_layer_buffer = []
        seen = set()
        for src in self.files:
            for fc in src.run_network.functions_calls:
                key = (fc.model_name, fc.layer_name)
                if key in seen:
                    continue
                seen.add(key)
                list_of_args.append(fc.args_cast)
                list_of_layer_buffer.append(fc.layer_buffer)

        cores0, cores1 = self.models_dedicated_cores[0], self.models_dedicated_cores[1]
        self.fused_content.append(combined_arg_cast_statement(list_of_args, list_of_layer_buffer, cores0, cores1))
        self.fused_content.append(combined_closure_call_statement(cores0, cores1))

        self.fused_content.append("\n}\n")


# ===== THESIS FIX (patch_multideeploy.py) =====
import re as _thesis_re


def rebase_core_ids(text: str, names, starts) -> str:
    """Inside each model's layer functions, absolute core ids become group-relative, so that a
    model placed on cores S..S+n-1 behaves as on cores 0..n-1 (Transpose, ReduceMean, ...)."""
    hdr = _thesis_re.compile(r"^static void __attribute__\(\(noinline\)\) (\w+)_layer_\d+\(", _thesis_re.M)
    heads = [(m.start(), m.group(1)) for m in hdr.finditer(text)]
    end_of_layers = text.find("static void Combined_closure_")
    out, last = [], 0
    for idx, (pos, name) in enumerate(heads):
        stop = heads[idx + 1][0] if idx + 1 < len(heads) else end_of_layers
        out.append(text[last:pos])
        body = text[pos:stop]
        S = starts[names.index(name)] if name in names else 0
        if S:
            body = body.replace("pi_core_id()", f"(pi_core_id() - {S})")
            body = _thesis_re.sub(r"\bBEGIN_SINGLE_CORE\b", f"if (pi_core_id() == {S}) {{", body)
            body = _thesis_re.sub(r"(?<!BEGIN_)(?<!END_)\bSINGLE_CORE\b", f"if (pi_core_id() == {S})", body)
        out.append(body)
        last = stop
    out.append(text[last:])
    return "".join(out)
