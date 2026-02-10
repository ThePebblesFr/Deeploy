import os
import sys
import re
from pathlib import Path
import json

from MultiDeeploy.WorkloadScheduler import WorkloadScheduler
from MultiDeeploy.ModelProfile import ModelProfile
from MultiDeeploy.CodeFusion.HeaderFusion import Network_h_fusion, testinputs_h_fusion, testoutputs_h_fusion
from MultiDeeploy.CodeFusion.SourceFile import SourceFile
from MultiDeeploy.CodeFusion.SourceFusion import SourceFusion

from MultiDeeployTest.testUtils.MultiDeeployArgParser import parse_arguments
from MultiDeeployTest.testUtils.UpdateMainDeeployTest import update_deeploy_test_main
from testUtils.UpdateMainDeeployTest import update_multideeploy_test_main

if __name__ == "__main__":

    args = parse_arguments()

    test_folder = args.test_folder
    models_name = args.models
    models_dedicated_cores = args.dedicated_cores
    models_l1_dedicated_size = args.dedicated_l1
    models_profiles = []

    # Run deeploy for the specified models
    for i in range(len(models_name)):

        update_deeploy_test_main(models_name[i])

        model_test_path = f"{test_folder}/{models_name[i]}"

        command = (
            f"python3 testRunner_tiled_siracusa.py "
            f"-t {model_test_path} "
            f"--defaultMemLevel L2 --profileTiling "
            f"--cores {models_dedicated_cores[i]} "
            f"--l1 {models_l1_dedicated_size[i]} "
            f"--plotMemAlloc "
            f"--name {models_name[i]} "
            f"--profileToJSON"
        )

        print(f"[MultiDeeploy] Simulation Command: {command}")

        err = os.system(command)
        if err != 0:
            print(f"[TestRunner] Error during the simulation of model {models_name[i]}")
            exit(1)

        json_profile_path = f"/app/Deeploy/MultiDeeployTest/TEST_SIRACUSA/{model_test_path}/deeployStates/{models_name[i]}_profiling.json"
        with open(json_profile_path) as f:
            data = json.load(f)

        model = ModelProfile.from_dict(data)
        models_profiles.append(model)

    workload_scheduler = WorkloadScheduler(models=models_profiles)
    workload_scheduler.run()
    workload_scheduler.print_schedule()

    # Channels renaming
    for model_name in models_name:
        model_network_c_path = Path(
            f"/app/Deeploy/MultiDeeployTest/TEST_SIRACUSA/{test_folder}/{model_name}/Network.c"
        )

        lines = model_network_c_path.read_text().splitlines(keepends=True)
        new_lines = []

        for line in lines:
            # Replace only whole identifiers
            line = line.replace(
                "channel_input",
                f"{model_name}_channel_input"
            )
            line = line.replace(
                "channel_output",
                f"{model_name}_channel_output"
            )
            new_lines.append(line)

        model_network_c_path.write_text("".join(new_lines))


    # Fuse Network.h files
    network_h_files = []
    for i in range(len(models_name)):
        model_test_path = f"{test_folder}/{models_name[i]}"
        network_h_path = Path(f"TEST_SIRACUSA/{model_test_path}/Network.h")
        network_h_files.append(network_h_path)

    output_network_h_path = Path(f"TEST_SIRACUSA/{test_folder}/Network.h")
    Network_h_fusion(files=network_h_files, output_path=output_network_h_path)

    # Fuse testinputs.h files
    testInputs_h_files = []
    for i in range(len(models_name)):
        model_test_path = f"{test_folder}/{models_name[i]}"
        testInputs_h_path = Path(f"TEST_SIRACUSA/{model_test_path}/testinputs.h")
        testInputs_h_files.append(testInputs_h_path)

    output_testInputs_h_path = Path(f"TEST_SIRACUSA/{test_folder}/testinputs.h")
    testinputs_h_fusion(files=testInputs_h_files, model_names=models_name, output_path=output_testInputs_h_path)

    # Fuse testoutputs.h files
    testOutputs_h_files = []
    for i in range(len(models_name)):
        model_test_path = f"{test_folder}/{models_name[i]}"
        testOutputs_h_path = Path(f"TEST_SIRACUSA/{model_test_path}/testoutputs.h")
        testOutputs_h_files.append(testOutputs_h_path)

    output_testOutputs_h_path = Path(f"TEST_SIRACUSA/{test_folder}/testoutputs.h")
    testoutputs_h_fusion(files=testOutputs_h_files, model_names=models_name, output_path=output_testOutputs_h_path)

    # Fuse Network.c files
    network_c_files = []
    sourcefile_c_objects = []
    for i in range(len(models_name)):
        model_test_path = f"{test_folder}/{models_name[i]}"
        network_c_path = Path(f"TEST_SIRACUSA/{model_test_path}/Network.c")
        network_c_files.append(network_c_path)

        with open(network_c_path) as f:
            content = f.read()

            source_file = SourceFile(
                path=str(network_c_path),
                content=content,
                model_name=models_name[i],
                layers=models_profiles[i].layers,
            )

            source_file.getGlobals()
            source_file.getInitNetworkFunction()
            source_file.getRunNetworkFunction()
            # print(f"RunNework functionCalls: {source_file.run_network.functions_calls}")
            source_file.getFunctions()
            # print(source_file.functions.layers_functions_definition[1].closure_body[1])

            sourcefile_c_objects.append(source_file)

    source_fusion = SourceFusion(
        files=sourcefile_c_objects,
        workload_scheduler=workload_scheduler,
        output_path=Path(f"TEST_SIRACUSA/{test_folder}/Network.c"),
        models_dedicated_cores=models_dedicated_cores
    )
    source_fusion.fuse()

    # Update the multideeploytest.c file
    update_multideeploy_test_main(models_name)

    # Run the fused multi-deeploy test
    test_name = test_folder.split("/")[-2]

    # CMake configuration for Multi-Deeploy test
    command = "$CMAKE -D TOOLCHAIN=LLVM -D TOOLCHAIN_INSTALL_DIR=/app/install/llvm -D GENERATED_SOURCE=/app/Deeploy/MultiDeeployTest/TEST_SIRACUSA/" + test_folder + " -D platform=Siracusa  -D NUM_CORES=8 -D banshee_simulation=OFF -D gvsoc_simulation=ON -B TEST_SIRACUSA/build -DENABLE_MULTIDEEPLOY_TEST=ON -D TESTNAME=" + test_name + " .."
    print(f"[MultiDeeploy] CMake configuration command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[TestRunner] Error during the cmake configuration of multi-deeploy test {test_name}")
        exit(1)

    # Build Multi-Deeploy test
    command = "$CMAKE --build /app/Deeploy/MultiDeeployTest/TEST_SIRACUSA/build --target " + test_name
    print(f"[MultiDeeploy] Build command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[TestRunner] Error during the build of multi-deeploy test {test_name}")
        exit(1)

    # Simulation of Multi-Deeploy test
    command = "$CMAKE --build TEST_SIRACUSA/build --target gvsoc_" + test_name
    print(f"[MultiDeeploy] Simulation command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[TestRunner] Error during the simulation of multi-deeploy test {test_name}")
        exit(1)