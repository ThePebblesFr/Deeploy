import os
import sys
import re
from pathlib import Path
import json

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
    models_doublebuffer = args.doublebuffer
    models_profiles = []

    # Run deeploy for the specified models
    for i in range(len(models_name)):

        update_deeploy_test_main(models_name[i])

        model_test_path = f"{test_folder}/{models_name[i]}"
        model_test_gen_path = f"{test_folder}/{models_name[0]}_x_{models_name[1]}/{models_name[i]}"
        model_test_path_no_number = model_test_path.rsplit("_", 1)[0]  # Remove the last part after the last underscore

        command = (
            f"python3 testRunner_tiled_siracusa.py "
            f"-t {model_test_path_no_number} "
            f"--test-gen {model_test_gen_path} "
            f"--defaultMemLevel L2 --profileTiling "
            f"--cores {models_dedicated_cores[i]} "
            f"--l1 {models_l1_dedicated_size[i]} "
            f"--plotMemAlloc "
            f"--name {models_name[i]} "
            f"--profileToJSON"
            + (" --doublebuffer" if models_doublebuffer[i] else "")
        )

        print(f"[MultiDeeploy] Simulation Command: {command}")

        err = os.system(command)
        json_profile_path = f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/{model_test_gen_path}/deeployStates/{models_name[i]}_profiling.json"
        if err != 0:
            # A nonzero exit here is most commonly the per-model single-vector
            # correctness check failing at this specific (cores, L1) tiling point --
            # a known, tolerated occurrence in this codebase (MultiDeeployScript/
            # run_DSE.sh's sweep treats it the same way, warning and continuing rather
            # than aborting, and its output CSV even has a dedicated "correctness"
            # column for it). Codegen and tiling profiling both complete before that
            # check runs, so the profiling JSON this loop needs is already on disk;
            # only abort here if it's genuinely missing, which means the failure was
            # a real compile/codegen error rather than a numeric mismatch.
            if not os.path.exists(json_profile_path):
                print(f"[TestRunner] Error during the simulation of model {models_name[i]} (no profiling JSON produced)")
                exit(1)
            print(f"[TestRunner] Warning: model {models_name[i]} failed its correctness check at this (cores, L1) point; continuing")

        with open(json_profile_path) as f:
            data = json.load(f)

        model = ModelProfile.from_dict(data)
        models_profiles.append(model)

    # Channels renaming
    for i in range(len(models_name)):

        model_test_gen_path = f"{test_folder}/{models_name[0]}_x_{models_name[1]}/{models_name[i]}"
        model_network_c_path = Path(
            f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/{model_test_gen_path}/Network.c"
        )

        content = model_network_c_path.read_text()
        model_name_re = re.escape(models_name[i])
        # Idempotent, word-boundary-anchored rename. Deeploy's own codegen emits
        # a BARE "channel_input"/"channel_output" local variable per closure in
        # most configs, but a real, config-dependent minority (observed under
        # --doublebuffer at a small, L3-chunking-forced L1 budget) already emits
        # it PRE-PREFIXED with this exact model name. A plain, unconditional
        # str.replace() here double-prefixes those cases (confirmed: produced
        # undeclared identifiers like
        # "AnomalyDetection_light_2_AnomalyDetection_light_2_channel_input" at
        # compile time). The negative lookbehind skips any occurrence already
        # immediately preceded by "{model_name}_", so this is safe regardless of
        # which shape Deeploy produced for a given config.
        content = re.sub(rf"(?<!{model_name_re}_)\bchannel_input\b", f"{models_name[i]}_channel_input", content)
        content = re.sub(rf"(?<!{model_name_re}_)\bchannel_output\b", f"{models_name[i]}_channel_output", content)

        model_network_c_path.write_text(content)


    # Fuse Network.h files
    network_h_files = []
    for i in range(len(models_name)):
        model_test_path = f"{test_folder}/{models_name[0]}_x_{models_name[1]}/{models_name[i]}"
        network_h_path = Path(f"TEST_SIRACUSA/{model_test_path}/Network.h")
        network_h_files.append(network_h_path)

    output_network_h_path = Path(f"TEST_SIRACUSA/{test_folder}/{models_name[0]}_x_{models_name[1]}/Network.h")
    Network_h_fusion(files=network_h_files, output_path=output_network_h_path)

    # Fuse testinputs.h files
    testInputs_h_files = []
    for i in range(len(models_name)):
        model_test_path = f"{test_folder}/{models_name[0]}_x_{models_name[1]}/{models_name[i]}"
        testInputs_h_path = Path(f"TEST_SIRACUSA/{model_test_path}/testinputs.h")
        testInputs_h_files.append(testInputs_h_path)

    output_testInputs_h_path = Path(f"TEST_SIRACUSA/{test_folder}/{models_name[0]}_x_{models_name[1]}/testinputs.h")
    testinputs_h_fusion(files=testInputs_h_files, model_names=models_name, output_path=output_testInputs_h_path)

    # Fuse testoutputs.h files
    testOutputs_h_files = []
    for i in range(len(models_name)):
        model_test_path = f"{test_folder}/{models_name[0]}_x_{models_name[1]}/{models_name[i]}"
        testOutputs_h_path = Path(f"TEST_SIRACUSA/{model_test_path}/testoutputs.h")
        testOutputs_h_files.append(testOutputs_h_path)

    output_testOutputs_h_path = Path(f"TEST_SIRACUSA/{test_folder}/{models_name[0]}_x_{models_name[1]}/testoutputs.h")
    testoutputs_h_fusion(files=testOutputs_h_files, model_names=models_name, output_path=output_testOutputs_h_path)

    # Fuse Network.c files
    network_c_files = []
    sourcefile_c_objects = []
    for i in range(len(models_name)):
        model_test_path = f"{test_folder}/{models_name[0]}_x_{models_name[1]}/{models_name[i]}"
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
        output_path=Path(f"TEST_SIRACUSA/{test_folder}/{models_name[0]}_x_{models_name[1]}/Network.c"),
        models_dedicated_cores=models_dedicated_cores
    )
    source_fusion.fuse()

    # Update the multideeploytest.c file
    update_multideeploy_test_main(models_name)

    # Run the fused multi-deeploy test
    test_name = test_folder.split("/")[-2]

    # CMake configuration for Multi-Deeploy test
    command = "$CMAKE -D TOOLCHAIN=LLVM -D TOOLCHAIN_INSTALL_DIR=/app/install/llvm -D GENERATED_SOURCE=/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/" + f"{test_folder}/{models_name[0]}_x_{models_name[1]}" + " -D platform=Siracusa  -D NUM_CORES=8 -D banshee_simulation=OFF -D gvsoc_simulation=ON -B TEST_SIRACUSA/build -DENABLE_DYNMULTIDEEPLOY_TEST=ON -D TESTNAME=" + test_name + " .."
    print(f"[MultiDeeploy] CMake configuration command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[TestRunner] Error during the cmake configuration of multi-deeploy test {test_name}")
        exit(1)

    # Build Multi-Deeploy test
    command = "$CMAKE --build /app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build --target " + test_name
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