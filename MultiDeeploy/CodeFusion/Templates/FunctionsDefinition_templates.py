from MultiDeeploy.utils import replace_num_cores


def dedup_lines_preserve_order(lines):
    if isinstance(lines, str):
        split = True
        lines_list = lines.splitlines(keepends=True)
    else:
        split = False
        lines_list = lines

    seen = set()
    out = []
    for line in lines_list:
        key = line.strip()
        if key == "":
            # keep blank lines as-is (optional)
            out.append(line)
            continue
        if key not in seen:
            seen.add(key)
            out.append(line)

    return "".join(out) if split else out

def struct_definition_statement(id_round, tiles, func_name):
    ret_str = f"""typedef struct {{\n"""
    temp_str = ""
    for tile in tiles:
        if func_name == "tiling_closure":
            temp_str += f"{tile.tiling_closure[0][1]}\n"
        temp_str += f"{tile.closure[0][1]}\n"
    ret_str += dedup_lines_preserve_order(temp_str)
    ret_str += f"""}} Round_{id_round}_{func_name}_args_t;\n"""
    return ret_str

def closure_function_definition_statement(id_round, tiles):
    ret_str = f"""static void Round_{id_round}_closure(Round_{id_round}_closure_args_t* Round_{id_round}_closure_args) {{\n"""
    # Args casting
    ret_str += f"""    Round_{id_round}_closure_args_t* args = (Round_{id_round}_closure_args_t*) Round_{id_round}_closure_args;\n"""
    temp_str = ""
    for tile in tiles:
        args_cast = tile.closure_body[0]
        temp_str += "".join(args_cast.splitlines(keepends=True))
    ret_str += dedup_lines_preserve_order(temp_str)
    ret_str += f"""\n\n"""

    # Arena allocations
    temp_str = ""
    for tile in tiles:
        temp_str += tile.closure_body[1]["buffers_init"]
    ret_str += dedup_lines_preserve_order(temp_str)
    ret_str += f"""\n\n"""

    # Tiling closure args casting
    ret_str += f"""    Round_{id_round}_tiling_closure_args_t Round_{id_round}_tiling_closure_args = (Round_{id_round}_tiling_closure_args_t) {{\n"""
    temp_str = ""
    for tile in tiles:
        temp_str += tile.closure_body[1]["tiling_closure_arg_cast_args"].rstrip("\n") + ",\n"
        temp_str += tile.closure_L3_body[1][1].rstrip("\n") + ",\n"
    ret_str += dedup_lines_preserve_order(temp_str)
    ret_str = ret_str.rstrip(",\n") + "};\n"

    # Tiling closure call
    ret_str += f"""    pi_cl_team_fork(NUM_CORES, (void *)Round_{id_round}_tiling_closure, &Round_{id_round}_tiling_closure_args);\n"""
    ret_str += f"""}}\n"""
    return ret_str

def generate_channels_declaration(model_name):
    ret_str = f"""
        uint32_t {model_name}_channel_input = (uint32_t)-1;
        uint32_t {model_name}_channel_output = (uint32_t)-1;

    """
    return ret_str


def tiling_closure_function_definition_statement(id_round, lf_map, tiles, tiles_by_model, models_dedicated_cores, ranges, new_layer_keys_of_the_round):

    ret_str = f"""static void Round_{id_round}_tiling_closure(Round_{id_round}_tiling_closure_args_t* Round_{id_round}_tiling_closure_args) {{\n"""
    # Args casting
    ret_str += f"""    Round_{id_round}_tiling_closure_args_t* args = (Round_{id_round}_tiling_closure_args_t*) Round_{id_round}_tiling_closure_args;\n"""
    temp_str = ""
    for tile in tiles:
            args_cast = tile.tiling_closure_body[0]
            temp_str += "".join(args_cast.splitlines(keepends=True))
            closure_args_cast = tile.closure_body[0]
            temp_str += "".join(closure_args_cast.splitlines(keepends=True))
    ret_str += dedup_lines_preserve_order(temp_str)
    ret_str += f"""\n\n"""

    new_keys = new_layer_keys_of_the_round
    if new_keys:
        for (mname, lname) in new_keys:

            tile_def = lf_map[(mname, lname)]
            for p in tile_def.closure_body[1]["pointers_init"]:
                init_stmt = f"{p[1].strip()}{p[2].strip()}\n"  # var + initializer
                ret_str += init_stmt

    ret_str += f"""

        int core_id = pi_core_id();
        uint32_t temp_start, temp_end;
    """

    nb_models = len(tiles_by_model)
    round_ranges = ranges[id_round]
    nb_tiles_done = 0

    ret_str += f"if (core_id == 0) {{\n"
    for model_idx in range(nb_models):
        ret_str += f"   subgroup_barrier_init({model_idx+2}, cores_map[{model_idx}][0], cores_map[{model_idx}][1]);\n"
    ret_str += "}\n"
    ret_str += f"   subgroup_barrier_init({nb_models+2}, 0, NUM_CORES - 1);\n"
    ret_str += f"""
        pi_cl_team_barrier();
        eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);
    """

    for model_idx in range(nb_models):
        model_tiles = tiles_by_model[model_idx]
        if not model_tiles:
            continue

        model_name = model_tiles[0][0]
        model_ranges = round_ranges.get(model_name, {})
        if not model_ranges:
            continue

        ret_str += generate_channels_declaration(model_name)


        ret_str += f"    if (cores_map[{model_idx}][0] <= core_id && core_id <= cores_map[{model_idx}][1]) {{\n"
        ret_str += f"""uint32_t nb_cycles_start_{model_name} = getCycles();\n"""
        # ret_str += f"""printf("Core %d executing {model_name} Round_{id_round}_tiling_closure started at %d cycles\\n", core_id, nb_cycles_start_{model_name});\n"""

        for layer_name, (start_i, end_i) in model_ranges.items():
            tile_def = lf_map.get((model_name, layer_name))

            ret_str += f"      for (int TILING_I = {start_i}; TILING_I < {end_i}; TILING_I++) {{\n"

            tile_body = tile_def.tiling_closure_body[1]
            tile_body = replace_num_cores(tile_body, models_dedicated_cores[model_idx])

            # DMA IN
            ret_str += f"        if (core_id == cores_map[{model_idx}][0]) {{\n"
            ret_str += f"            temp_start = getCycles();\n"
            ret_str += tile_def.closure_body[1]["dma_in_transfer"]
            ret_str += tile_def.closure_body[1]["pre_cluster_fork_setup"]
            # ret_str += "printf(\"Core %d finished DMA in " + layer_name + " TILING_I=%d\\n\", core_id, TILING_I);\n"
            ret_str += f"            temp_end = getCycles();\n"
            ret_str += f"            round_{id_round}_tiles_timings[{nb_tiles_done - start_i} + TILING_I][0] = temp_end - temp_start;\n"
            ret_str += f"            temp_start = getCycles();\n"
            ret_str += "        }\n"
            ret_str += f"        subgroup_barrier_wait({model_idx+2}, core_id);\n"

            # KERNEL EXEC
            ret_str += tile_body
            # ret_str += "printf(\"Core %d finished kernel " + layer_name + " TILING_I=%d\\n\", core_id, TILING_I);\n"
            ret_str += f"        subgroup_barrier_wait({model_idx+2}, core_id);\n"

            # DMA OUT
            ret_str += f"        if (core_id == cores_map[{model_idx}][0]) {{\n"
            ret_str += f"            temp_end = getCycles();\n"
            ret_str += f"            round_{id_round}_tiles_timings[{nb_tiles_done - start_i} + TILING_I][1] = temp_end - temp_start;\n"
            ret_str += f"            temp_start = getCycles();\n"
            ret_str += tile_def.closure_body[1]["dma_out_transfer"]
            ret_str += tile_def.closure_body[1]["dma_out_getCycles_end"]
            # ret_str += "printf(\"Core %d finished DMA out " + layer_name + " TILING_I=%d\\n\", core_id, TILING_I);\n"
            ret_str += f"            temp_end = getCycles();\n"
            ret_str += f"            round_{id_round}_tiles_timings[{nb_tiles_done - start_i} + TILING_I][2] = temp_end - temp_start;\n"
            ret_str += "        }\n"
            ret_str += f"        subgroup_barrier_wait({model_idx+2}, core_id);\n"

            ret_str += "      }\n"
            nb_tiles_done += end_i - start_i

        ret_str += f"""uint32_t nb_cycles_{layer_name}_{model_name} = getCycles();\n"""
        # ret_str += f"""printf("Core %d finished layer {layer_name} of model {model_name} in %d cycles\\n", core_id, nb_cycles_{layer_name}_{model_name});\n"""

        ret_str += "    }\n"

    ret_str += f"       subgroup_barrier_wait({nb_models+2}, core_id);\n"
    # ret_str += "        pi_cl_team_barrier();\n"
    ret_str += "}\n"
    return ret_str