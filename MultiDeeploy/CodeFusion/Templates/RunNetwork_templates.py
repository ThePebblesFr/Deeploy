from MultiDeeploy.CodeFusion.Templates.FunctionsDefinition_templates import dedup_lines_preserve_order


def arg_cast_statement(id_round, args, layer_buffers) -> str:
    ret_str = ""
    for buf in layer_buffers:
        ret_str += f"{buf}\n"
    ret_str += f"""Round_{id_round}_closure_args_t Round_{id_round}_closure_args = (Round_{id_round}_closure_args_t) {{\n"""
    temp_str = ""
    for arg in args:
        temp_str += f"{arg},\n"
    ret_str += dedup_lines_preserve_order(temp_str)
    ret_str += "};\n"
    return ret_str

def closure_call_statement(id_round):
    return f"""    Round_{id_round}_closure(&Round_{id_round}_closure_args);\n"""