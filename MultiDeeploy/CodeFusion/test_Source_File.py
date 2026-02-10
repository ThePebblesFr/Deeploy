from MultiDeeploy.CodeFusion.SourceFile import SourceFile

with open("/app/Deeploy/MultiDeeployTest/TEST_SIRACUSA/Tests/miniMNV2_x_AnomalyDetection/miniMNV2/Network.c") as f:
    content = f.read()

    source_file = SourceFile(
        path="/app/Deeploy/MultiDeeployTest/TEST_SIRACUSA/Tests/miniMNV2_x_AnomalyDetection/miniMNV2/Network.c",
        content=content,
        model_name="miniMNV2"
    )

    source_file.getGlobals()
    source_file.getInitNetworkFunction()
    source_file.getRunNetworkFunction()

    # print("---- GLOBALS ----")
    # for line in source_file.globals.content:
    #     print(line, end="")

    # print("\n---- INIT NETWORK FUNCTION ----")
    # for line in source_file.init_network.content:
    #     print(line, end="")
    
    print("\n---- RUN NETWORK FUNCTION CALLS ----")
    # for line in source_file.run_network.variables:
    #     print(line, end="")

    for func_call in source_file.run_network.functions_calls:
        print("---- FUNCTION CALL ----")
        print(f"--> layer buffer:\n{func_call.layer_buffer}")
        print(f"--> prefix args cast:\n{func_call.prefix_args_cast}")
        print(f"--> args cast:\n{func_call.args_cast}")
        print(f"--> layer function call:\n{func_call.layer_function_call}")