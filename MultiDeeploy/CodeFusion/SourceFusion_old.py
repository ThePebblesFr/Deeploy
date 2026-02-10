from typing import List
from MultiDeeploy.CodeFusion.SourceFile import SourceFile
from MultiDeeploy.WorkloadScheduler import WorkloadScheduler
from MultiDeeploy.utils import replace_num_cores

class SourceFusion():

    def __init__(self, files: List[SourceFile], workload_scheduler: WorkloadScheduler, output_path: str, models_dedicated_cores: List[int]):
        self.files = files
        self.workload_scheduler = workload_scheduler
        self.output_path = output_path
        self.models_dedicated_cores = models_dedicated_cores,
        self.fused_content: List[str] = []

    def fuse(self) -> None:
        self.globals_fusion()
        self.functions_fusion()
        self.run_network_fusion()
        self.init_network_fusion()

        with open(self.output_path, 'w') as f:
            f.writelines(self.fused_content)

    def globals_fusion(self) -> None:
        seen_lines = set()
        cluster_dev_added = False

        for source in self.files:
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

        self.fused_content.append("\n")

        # Adding cores_map definition
        nb_models = len(self.files)
        self.fused_content.append(f"int cores_map[{nb_models}][2] = {{")
        core_cursor = 0
        for i in range(nb_models):
            nb_cores = self.models_dedicated_cores[0][i]
            start = core_cursor
            end = core_cursor + nb_cores - 1
            self.fused_content.append(f"{{{start}, {end}}}")
            core_cursor += nb_cores
            if i < nb_models - 1:
                self.fused_content.append(", ")
        self.fused_content.append("};\n")

        self.fused_content.append(f"int nb_dedicated_cores[{nb_models}] = {{")
        for i in range(nb_models):
            self.fused_content.append(str(self.models_dedicated_cores[0][i]))
            if i < nb_models - 1:
                self.fused_content.append(", ")

        self.fused_content.append("};\n")

    def functions_fusion(self) -> None:
        """
        Fuse function definitions (cluster_fork + closure_L3 + closure) according to the scheduler.

        Strategy:
        - Build a map (model_name, layer_name) -> LayerFunctionDefinition
            by aligning:
            * unique logical layer order from src.run_network.functions_calls (fc.layer_name)
            * src.functions.layers_functions_definition (same order in the C file)

        - For each scheduling round:
            * collapse tiles to unique (model_name, layer_name) keys (once per logical layer per model per round)
            * find the round's "longest layer" (scheduler)
            * emit fused code for:
                - tiling_closure
                - closure_L3
                - closure
                - cluster_fork

        """

        lf_map = {}

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

            for layer_name, layer_def in zip(ordered, layer_defs):
                key = (src.model_name, layer_name)
                if key in lf_map:
                    raise RuntimeError(f"Duplicate LayerFunctionDefinition for {key}")
                lf_map[key] = layer_def

        # ----------------------------
        # Helpers
        # ----------------------------
        def _unique_round_keys(round_layers):
            keys = []
            seen_local = set()
            for model_tiles in round_layers:
                for tile in model_tiles:
                    k = (tile.model_name, tile.layer_name)
                    if k not in seen_local:
                        seen_local.add(k)
                        keys.append(k)
            return keys

        def _first_layer_per_model(round_layers):
            out = {}
            for model_tiles in round_layers:
                if model_tiles:
                    t = model_tiles[0]
                    out[t.model_name] = (t.model_name, t.layer_name)
            return out


        def _last_layer_per_model(round_layers):
            out = {}
            for model_tiles in round_layers:
                if model_tiles:
                    t = model_tiles[-1]
                    out[t.model_name] = (t.model_name, t.layer_name)
            return out

        def _drop_first_line(s: str) -> str:
            idx = s.find(";")
            if idx == -1:
                return ""
            return s[idx + 1 :]
        
        def _dedup_lines_preserve_order(lines):
            if isinstance(lines, str):
                split = True
                lines_list = lines.splitlines(keepends=True)
            else:
                split = False
                lines_list = lines
            seen = set()
            out = []
            for line in lines_list:
                if line not in seen:
                    seen.add(line)
                    out.append(line)
            return "".join(out) if split else out

        def _emit_fused(kind: str, round_idx: int, round_keys, longest_key):
            longest_def = lf_map[longest_key]

            if kind == "closure_L3":
                block = longest_def.closure_L3
                body  = longest_def.closure_L3_body
                body_attr = "closure_L3_body"
            elif kind == "cluster_fork":
                block = longest_def.cluster_fork
                body  = longest_def.cluster_fork_body
                body_attr = "cluster_fork_body"
            else:
                raise ValueError(f"Unknown kind: {kind}")

            self.fused_content.append(
                f"\n// ===== FUSED {kind} - Scheduling Round {round_idx} =====\n"
            )

            # -------------------------------------------------
            # STRUCT FUSION
            # -------------------------------------------------
            struct_prefix = block[0][0]
            struct_suffix = block[0][2]

            temp_added_content = [struct_prefix]

            for k in round_keys:
                layer_def = lf_map.get(k)
                layer_block = getattr(layer_def, kind)
                temp_added_content.append(layer_block[0][1])

                if kind == "cluster_fork":
                    temp_added_content.append(layer_def.closure[0][1])

            self.fused_content.append(
                _dedup_lines_preserve_order("".join(temp_added_content))
            )
            self.fused_content.append(struct_suffix)

            # -------------------------------------------------
            # CALL PREFIX
            # -------------------------------------------------
            call_prefix = block[1][0]
            call_suffix = block[1][2]
            self.fused_content.append(call_prefix)

            # -------------------------------------------------
            # ARG CAST
            # -------------------------------------------------
            temp_added_content = [body[0]]

            if kind == "cluster_fork":
                temp_added_content.append(
                    _drop_first_line(longest_def.closure_body[0])
                )

            for k in round_keys:
                if k == longest_key:
                    continue
                layer_def = lf_map[k]
                layer_body = getattr(layer_def, body_attr)
                temp_added_content.append(_drop_first_line(layer_body[0]))

                if kind == "cluster_fork":
                    temp_added_content.append(
                        _drop_first_line(layer_def.closure_body[0])
                    )

            self.fused_content.append(_dedup_lines_preserve_order("".join(temp_added_content)))

            # -------------------------------------------------
            # COMPOUND LITERAL
            # -------------------------------------------------
            temp_added_content = [body[1][0]]  # prefix

            for k in round_keys:
                layer_def = lf_map[k]
                layer_body = getattr(layer_def, body_attr)
                temp_added_content.append(layer_body[1][1][:-1] + ",\n")

                if kind == "cluster_fork":
                    temp_added_content.append(layer_def.closure_L3_body[1][1][:-1] + ",\n")

            # remove last comma
            temp_added_content[-1] = temp_added_content[-1].rstrip(",")
            temp_added_content.append(body[1][2])  # suffix

            self.fused_content.append(_dedup_lines_preserve_order("".join(temp_added_content)))

            # -------------------------------------------------
            # CALL
            # -------------------------------------------------
            self.fused_content.append(body[2])
            self.fused_content.append(call_suffix)


        def _dedup_keys_preserve_order(keys):
            seen = set()
            out = []
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
            return out

        def _tile_key(tile) -> tuple[str, str]:
            return (tile.model_name, tile.layer_name)
            

        # ----------------------------
        # 2) Emit per round (tiling_closure / cluster_fork / closure / closure_L3)
        # ----------------------------
        for round_idx, round_layers in enumerate(self.workload_scheduler.Scheduling_queue):

            longest_model_idx = self.workload_scheduler.Longest_layer_model_idx_per_round[round_idx]
            longest_tile = round_layers[longest_model_idx][0]
            longest_key = (longest_tile.model_name, longest_tile.layer_name)
            round_keys = _unique_round_keys(round_layers)
            
            # -------------------------------------------------
            # tiling_closure fusion
            # -------------------------------------------------
            nb_models = len(round_layers)
            if longest_key not in lf_map:
                raise RuntimeError(f"Longest tile not found in lf_map: {longest_key}")

            longest_def = lf_map[longest_key]

            # --- per-round tiles (NO dedup here: you want per-tile emission) ---
            tiles_by_model: list[list[tuple[str, str]]] = []
            for i in range(nb_models):
                tiles_by_model.append([_tile_key(t) for t in round_layers[i]])

            # --- all logical layer keys involved this round, dedup for struct/arg_cast ---
            logical_keys = _dedup_keys_preserve_order(
                [k for model_tiles in tiles_by_model for k in model_tiles]
            )

            self.fused_content.append(
                f"\n// ===== FUSED tiling_closure – Scheduling Round {round_idx} =====\n"
            )

            # 1) STRUCT FUSION (dedup)
            struct_prefix, _, struct_suffix = longest_def.tiling_closure[0]

            temp_added_content = [struct_prefix]
            for k in logical_keys:
                temp_added_content.append(lf_map[k].tiling_closure[0][1])  # struct args
                temp_added_content.append(lf_map[k].closure[0][1]) # also add closure struct args
            temp_added_content.append(struct_suffix)  # only one suffix, longest
            self.fused_content.append(_dedup_lines_preserve_order("".join(temp_added_content)))

            # 2) function prefix (longest)
            call_prefix, _, _ = longest_def.tiling_closure[1]
            self.fused_content.append(call_prefix)

            # 3) ARG CAST FUSION (dedup; drop 1st line for non-longest)
            longest_arg_cast = longest_def.tiling_closure_body[0]
            temp_added_content = [longest_arg_cast]

            for k in logical_keys:
                if k == longest_key:
                    continue
                other_arg_cast = lf_map[k].tiling_closure_body[0]
                temp_added_content.append(_drop_first_line(other_arg_cast))
                other_closure_arg_cast = lf_map[k].closure_body[0]
                temp_added_content.append(_drop_first_line(other_closure_arg_cast))
            
            # # 4) pointers_init for each tile that is NOT the 1st tile of its model this round
            for model_idx in range(nb_models):
                model_tiles = tiles_by_model[model_idx]
                for j, k in enumerate(model_tiles):
                    if j == 0:
                        continue  # skip first tile
                    temp_added_content.append(lf_map[k].closure_body[1]["pointers_init"])
            self.fused_content.append(_dedup_lines_preserve_order("".join(temp_added_content)))

            # 5) DMA init of longest layer  + core_id
            self.fused_content.append(longest_def.closure_body[1]["dma_init"])
            self.fused_content.append("  int core_id = pi_core_id();\n")

            # 6) Per-model dispatch via cores_map
            #
            #   if (cores_map[i][0] <= core_id && core_id < cores_map[i][1]) {
            #       for each tile t:
            #         if first tile: function_body(t)
            #         else:
            #            DMA_out, ptr_incr, DMA_in, precluster_setup only on cores_map[i][0]
            #            function_body(t)
            #   }
            for model_idx in range(nb_models):
                model_tiles = tiles_by_model[model_idx]
                if not model_tiles:
                    continue

                self.fused_content.append(
                    f"  if (cores_map[{model_idx}][0] <= core_id && core_id < cores_map[{model_idx}][1]) {{\n"
                )

                for tile_j, k in enumerate(model_tiles):
                    tile_def = lf_map[k]
                    tile_body = tile_def.tiling_closure_body[1]

                    if tile_j == 0:
                        # First tile → direct body
                        tile_body = replace_num_cores(tile_body, self.models_dedicated_cores[0][model_idx])
                        self.fused_content.append(tile_body)
                    else:
                        # Non-first tile → DMA on leader core only
                        self.fused_content.append(
                            f"    if (core_id == cores_map[{model_idx}][0]) {{\n"
                        )
                        self.fused_content.append(tile_def.closure_body[1]["dma_out_transfer"])
                        self.fused_content.append(tile_def.closure_body[1]["end_for_loop_and_ptr_increment"])
                        self.fused_content.append(tile_def.closure_body[1]["dma_in_transfer"])
                        self.fused_content.append(tile_def.closure_body[1]["pre_cluster_fork_setup"])
                        self.fused_content.append("    }\n")

                        tile_body = replace_num_cores(tile_body, self.models_dedicated_cores[0][model_idx])
                        self.fused_content.append(tile_body)


                self.fused_content.append("  }\n")

            # 7) Close the C function
            self.fused_content.append("}\n")

            # -------------------------------------------------
            # cluster_fork fusion
            # -------------------------------------------------
            _emit_fused("cluster_fork", round_idx, round_keys, longest_key)

            # -------------------------------------------------
            # closure fusion
            # -------------------------------------------------
            longest_def = lf_map[longest_key]

            round_keys = _unique_round_keys(round_layers)
            first_per_model = _first_layer_per_model(round_layers)
            last_per_model = _last_layer_per_model(round_layers)

            self.fused_content.append(
                f"\n// ===== FUSED closure – Scheduling Round {round_idx} =====\n"
            )

            # STRUCT FUSION
            struct_prefix, _, struct_suffix = longest_def.closure[0]

            temp_added_content = [struct_prefix]

            for k in round_keys:
                temp_added_content.append(lf_map[k].closure[0][1])
            temp_added_content.append(struct_suffix)
            self.fused_content.append(_dedup_lines_preserve_order("".join(temp_added_content)))

            # CALL PREFIX
            call_prefix, _, _ = longest_def.closure[1]
            self.fused_content.append(call_prefix)


            # ARG CAST (closure_body[0])
            longest_args_cast = longest_def.closure_body[0]
            temp_added_content = [longest_args_cast]

            for k in round_keys:
                if k == longest_key:
                    continue
                other_args_cast = lf_map[k].closure_body[0]
                # drop first line to avoid duplicate arg struct cast
                temp_added_content.append("".join(other_args_cast.splitlines(keepends=True)[2:]))
                other_closure_arg_cast = lf_map[k].closure_body[0]
                temp_added_content.append(_drop_first_line(other_closure_arg_cast))
            
            self.fused_content.append(_dedup_lines_preserve_order("".join(temp_added_content)))

            # BODY FUSION (closure_body[1])
            body = longest_def.closure_body[1]

            # buffers_init
            for k in round_keys:
                self.fused_content.append(lf_map[k].closure_body[1]["buffers_init"])

            # pointers_init
            temp_added_content = []
            for k in first_per_model.values():
                temp_added_content.append(lf_map[k].closure_body[1]["pointers_init"])
            for k in last_per_model.values():
                temp_added_content.append(lf_map[k].closure_body[1]["pointers_init"])
            self.fused_content.append(_dedup_lines_preserve_order("".join(temp_added_content)))

            # measurements + dma init (longest only)
            self.fused_content.append(body["measurements_init"])
            self.fused_content.append(body["dma_init"])

            # loop + ingress
            self.fused_content.append(body["for_loop_statement"])
            self.fused_content.append(body["dma_in_getCycles_start"])

            # DMA IN transfer → first layer per model
            for k in first_per_model.values():
                self.fused_content.append(
                    lf_map[k].closure_body[1]["dma_in_transfer"]
                )

            self.fused_content.append(body["dma_in_getCycles_end"])
            self.fused_content.append(body["kernel_getCycles_start"])

            # UPDATE VARIABLE / pre-cluster setup → first layer per model
            for k in first_per_model.values():
                self.fused_content.append(
                    lf_map[k].closure_body[1]["pre_cluster_fork_setup"]
                )

            # CLUSTER FORK ARG CAST
            temp_added_content = [body["tiling_closure_arg_cast_prefix"]]

            for k in round_keys:
                temp_added_content.append(lf_map[k].closure_body[1]["tiling_closure_arg_cast_args"][:-1])
                temp_added_content.append(",\n")
                temp_added_content.append(lf_map[k].closure_L3_body[1][1][:-1])
                temp_added_content.append(",\n")
            temp_added_content[-1] = temp_added_content[-1][:-3]  # remove last comma
            temp_added_content.append(body["tiling_closure_arg_cast_suffix"])
            self.fused_content.append(_dedup_lines_preserve_order("".join(temp_added_content)))

            # CLUSTER FORK CALL + KERNEL END
            self.fused_content.append(body["pi_cl_team_fork_call"])
            self.fused_content.append(body["kernel_getCycles_end"])
            self.fused_content.append(body["dma_out_getCycles_start"])

            # DMA OUT transfer → last layer per model
            for k in last_per_model.values():
                self.fused_content.append(
                    lf_map[k].closure_body[1]["dma_out_transfer"]
                )

            self.fused_content.append(body["dma_out_getCycles_end"])
            self.fused_content.append("}\n")
            # end loop + ptr increment → every layer
            for k in round_keys:
                self.fused_content.append(
                    lf_map[k].closure_body[1]["end_for_loop_and_ptr_increment"]
                )

            # profiling → longest only
            self.fused_content.append(body["profiling_block"])

            # CLOSE FUNCTION
            self.fused_content.append("\n}\n")

            # -------------------------------------------------
            # closure_L3 fusion
            # -------------------------------------------------
            _emit_fused("closure_L3",   round_idx, round_keys, longest_key)
            

    def init_network_fusion(self) -> None:
        self.fused_content.append("\nvoid InitNetwork() {\n")

        for source in self.files:
            for line in source.init_network.content:
                self.fused_content.append(line)

        self.fused_content.append("}\n")

    def run_network_fusion(self) -> None:
        """
        Fuse RunNetwork() functions according to the WorkloadScheduler.
        Tiles are collapsed to logical layers PER ROUND.
        """

        fc_map = {}
        for src in self.files:
            for fc in src.run_network.functions_calls:
                key = (fc.model_name, fc.layer_name)
                if key in fc_map:
                    raise RuntimeError(f"Duplicate FunctionCall for {key}")
                fc_map[key] = fc

        self.fused_content.append("\nvoid RunNetwork() {\n")

        for src in self.files:
            for var_stmt in src.run_network.variables:
                self.fused_content.append(var_stmt)
        self.fused_content.append("\n")

        for round_idx, round_layers in enumerate(self.workload_scheduler.Scheduling_queue):

            longest_model_idx = self.workload_scheduler.Longest_layer_model_idx_per_round[round_idx]
            longest_tile = round_layers[longest_model_idx][0]
            longest_key = (longest_tile.model_name, longest_tile.layer_name)

            if longest_key not in fc_map:
                raise RuntimeError(f"Longest layer not found in FunctionCalls: {longest_key}")

            longest_fc = fc_map[longest_key]

            round_keys = []
            seen = set()
            for model_layers in round_layers:
                for tile in model_layers:
                    key = (tile.model_name, tile.layer_name)
                    if key not in seen:
                        if key not in fc_map:
                            raise RuntimeError(f"Scheduled layer not found in code: {key}")
                        seen.add(key)
                        round_keys.append(key)

            self.fused_content.append(
                f"\n  // ===== Scheduling Round {round_idx} =====\n"
            )

            for key in round_keys:
                fc = fc_map[key]
                if fc.layer_buffer:
                    self.fused_content.append(fc.layer_buffer)

            if longest_fc.prefix_args_cast:
                self.fused_content.append(longest_fc.prefix_args_cast)

            for key in round_keys:
                fc = fc_map[key]
                if fc.args_cast:
                    self.fused_content.append(fc.args_cast)
                    self.fused_content[-1] += ","
            self.fused_content[-1] = self.fused_content[-1][:-1]

            if longest_fc.layer_function_call:
                self.fused_content.append(longest_fc.layer_function_call)

        self.fused_content.append("\n}\n")
