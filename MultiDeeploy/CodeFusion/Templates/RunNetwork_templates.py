from MultiDeeploy.CodeFusion.Templates.FunctionsDefinition_templates import dedup_lines_preserve_order


def combined_arg_cast_statement(args, layer_buffers, cores0, cores1) -> str:
    ret_str = ""
    for buf in layer_buffers:
        ret_str += f"{buf}\n"
    ret_str += f"""Combined_closure_{cores0}_{cores1}_args_t Combined_closure_args = (Combined_closure_{cores0}_{cores1}_args_t) {{\n"""
    temp_str = ""
    for arg in args:
        temp_str += f"{arg},\n"
    ret_str += dedup_lines_preserve_order(temp_str)
    ret_str += "};\n"
    return ret_str


def combined_closure_call_statement(cores0, cores1) -> str:
    # Dispatched via pi_cl_team_fork (NOT a plain call), matching exactly how the
    # old round-based "closure" invoked its nested "tiling_closure" -- the
    # combined closure's own pi_cl_team_barrier() call (used to rendezvous all
    # cores after core 0 finishes subgroup_barrier_init, before any of them wait
    # on a subgroup barrier) needs a guaranteed-active team-fork context, and this
    # is the one already proven to provide it. Collapses the old two-level
    # "closure calls tiling_closure via fork" into one level: RunNetwork calls
    # Combined_closure via fork directly.
    return (
        f"""    pi_cl_team_fork(NUM_CORES, (void *)Combined_closure_{cores0}_{cores1}, """
        f"""&Combined_closure_args);\n"""
    )
