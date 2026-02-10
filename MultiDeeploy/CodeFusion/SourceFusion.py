from typing import List
from MultiDeeploy.CodeFusion.SourceFile import SourceFile
from MultiDeeploy.CodeFusion.Templates.FunctionsDefinition_templates import closure_function_definition_statement, struct_definition_statement, tiling_closure_function_definition_statement
from MultiDeeploy.WorkloadScheduler import WorkloadScheduler
from MultiDeeploy.utils import compute_tiling_i_ranges, generate_subgroup_barriers_code, normalize_args_cast, replace_num_cores
from MultiDeeploy.CodeFusion.Templates.RunNetwork_templates import arg_cast_statement, closure_call_statement

class SourceFusion():

    def __init__(self, files: List[SourceFile], workload_scheduler: WorkloadScheduler, output_path: str, models_dedicated_cores: List[int]):
        self.files = files
        self.workload_scheduler = workload_scheduler
        self.output_path = output_path
        self.models_dedicated_cores = models_dedicated_cores
        self.fused_content: List[str] = []
        self.new_layer_keys_by_round = []
        self.lf_map = {}

    def _compute_new_layer_keys_by_round(self):
        """
        Returns: list[set[(model_name, layer_name)]] where each element is
        the set of layers that appear for the first time in that round.
        """
        seen_global = set()
        new_by_round = []

        for round_layers in self.workload_scheduler.Scheduling_queue:
            new_this_round = set()
            for model_layers in round_layers:
                for t in model_layers:
                    key = (t.model_name, t.layer_name)
                    if key not in seen_global:
                        seen_global.add(key)
                        new_this_round.add(key)
            new_by_round.append(new_this_round)

        return new_by_round

    def fuse(self) -> None:
        self.new_layer_keys_by_round = self._compute_new_layer_keys_by_round()
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
            nb_cores = self.models_dedicated_cores[i]
            start = core_cursor
            end = core_cursor + nb_cores - 1
            self.fused_content.append(f"{{{start}, {end}}}")
            core_cursor += nb_cores
            if i < nb_models - 1:
                self.fused_content.append(", ")
        self.fused_content.append("};\n")

        self.fused_content.append(f"int nb_dedicated_cores[{nb_models}] = {{")
        for i in range(nb_models):
            self.fused_content.append(str(self.models_dedicated_cores[i]))
            if i < nb_models - 1:
                self.fused_content.append(", ")

        self.fused_content.append("};\n")
        self.fused_content.append(generate_subgroup_barriers_code())

        for i in range(nb_models):
            self.fused_content.append(f"PI_L1 static subgroup_barrier_t g_barrier_{i};\n")

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

        self.lf_map = lf_map
        ranges = compute_tiling_i_ranges(self.workload_scheduler.Scheduling_queue)
        emitted_global_ptr_decls = set()

        # ----------------------------
        # 2) Emit per round (tiling_closure / closure)
        # ----------------------------
        for round_idx, round_layers in enumerate(self.workload_scheduler.Scheduling_queue):
            round_keys = []
            seen = set()
            for model_layers in round_layers:
                for tile in model_layers:
                    key = (tile.model_name, tile.layer_name)
                    if key not in seen:
                        if key not in lf_map:
                            raise RuntimeError(f"[Round {round_idx}] Scheduled layer not found in parsed functions: {key}")
                        seen.add(key)
                        round_keys.append(key)

            # 2) Map those keys to LayerFunctionDefinition objects ("tiles" for this round)
            tiles = [lf_map[key] for key in round_keys]
            
            nb_models = len(round_layers)
            tiles_by_model = []
            tiles_by_model_keys = []

            for model_idx in range(nb_models):
                model_tiles_defs = []
                model_keys = []

                for t in round_layers[model_idx]:
                    k = (t.model_name, t.layer_name)
                    if k not in lf_map:
                        raise RuntimeError(f"[Round {round_idx}] Scheduled tile not found in lf_map: {k}")

                    model_keys.append(k)
                    model_tiles_defs.append(lf_map[k])  # keep order, no dedup

                tiles_by_model.append(model_tiles_defs)
                tiles_by_model_keys.append(model_keys)

            # 3) Emit the four artifacts you requested
            self.fused_content.append(f"\n// ===== FUSED FUNCTIONS: Scheduling Round {round_idx} =====\n\n")

            new_keys = self.new_layer_keys_by_round[round_idx]
            if new_keys:
                # Emit global pointer decls once (ever), only for first-time layers
                for (mname, lname) in new_keys:
                    tile_def = self.lf_map[(mname, lname)]
                    for p in tile_def.closure_body[1]["pointers_init"]:
                        # p = (type_str, var_name, initializer)
                        decl = f"{p[0].rstrip()}{p[1].strip()};\n"
                        if decl not in emitted_global_ptr_decls:
                            emitted_global_ptr_decls.add(decl)
                            self.fused_content.append(decl)

                self.fused_content.append("\n")


            # (a) struct for tiling_closure
            self.fused_content.append(struct_definition_statement(round_idx, tiles, "tiling_closure"))
            self.fused_content.append("\n")

            # (b) tiling_closure function
            self.fused_content.append(
                tiling_closure_function_definition_statement(
                    round_idx,
                    lf_map,
                    tiles,
                    tiles_by_model_keys,
                    self.models_dedicated_cores,
                    ranges,
                    self.new_layer_keys_by_round[round_idx]
                )
            )
            self.fused_content.append("\n")

            # (c) struct for closure
            self.fused_content.append(struct_definition_statement(round_idx, tiles, "closure"))
            self.fused_content.append("\n")

            # (d) closure function
            self.fused_content.append(closure_function_definition_statement(round_idx, tiles))
            self.fused_content.append("\n")
            
            

    def init_network_fusion(self) -> None:
        self.fused_content.append("\nvoid InitNetwork() {\n")

        for source in self.files:
            for line in source.init_network.content:
                self.fused_content.append(line)

        for i, source in enumerate(self.files):
            self.fused_content.append(f"subgroup_barrier_init(&g_barrier_{i}, cores_map[{i}][0], cores_map[{i}][1]);\n")

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
        self.fused_content.append("\n")

        for round_idx, round_layers in enumerate(self.workload_scheduler.Scheduling_queue):
            # new_keys = self.new_layer_keys_by_round[round_idx]
            # if new_keys:
            #     for (mname, lname) in new_keys:
            #         fc = fc_map[(mname, lname)]

            #         tile_def = self.lf_map[(mname, lname)]
            #         for p in tile_def.closure_body[1]["pointers_init"]:
            #             init_stmt = f"{p[1].strip()}{p[2].strip()}\n"  # var + initializer
            #             self.fused_content.append(init_stmt)

            #     self.fused_content.append("\n")

            self.fused_content.append(f"// ===== Scheduling Round {round_idx} =====\n")
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
        
            list_of_args = []
            list_of_layer_buffer = []
            for key in round_keys:
                fc = fc_map[key]
                list_of_args.append(fc.args_cast)
                list_of_layer_buffer.append(fc.layer_buffer)

            self.fused_content.append(arg_cast_statement(round_idx, list_of_args, list_of_layer_buffer))
            self.fused_content.append(closure_call_statement(round_idx))

        self.fused_content.append("\n}\n")
