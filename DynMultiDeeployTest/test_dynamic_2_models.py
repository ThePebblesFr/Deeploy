import os
import sys
import re
from pathlib import Path
import json

from testUtils.DynMultiDeeployArgParser import parse_arguments
from testUtils.UpdateMainDeeployTest import update_dynmultideeploy_test_main
from DynMultiDeeploy.HeaderFusion import fuse_variables, strip_core_suffix
from DynMultiDeeploy.SourceFusion import fuse_network_c

if __name__ == "__main__":

    args = parse_arguments()
    test_folder = args.test_folder
    models_name = args.models

    test_c_path = "/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/" + models_name[0] + "_x_" + models_name[1]
    cores_splits = [[2, 6], [4, 4], [6, 2]]
    l1_splits = [[16000, 48000], [32000, 32000], [48000, 16000]]

    if args.gen_step:

        # MultiDeeployTests generation

        for i, (g0, g1) in enumerate(cores_splits):
            # Create a folder for each configuration
            os.system(f"mkdir -p {test_c_path}/{models_name[0]}_{g0}_x_{models_name[1]}_{g1}/{models_name[0]}_{g0}")
            os.system(f"mkdir -p {test_c_path}/{models_name[0]}_{g0}_x_{models_name[1]}_{g1}/{models_name[1]}_{g1}")

            # Run MultiDeeploy for each configuration
            command = (
                f"python3 testRunner_2_models.py "
                f"-t {args.test_folder} "
                f"--models {models_name[0]}_{g0} {models_name[1]}_{g1} "
                f"--dedicated-cores {g0} {g1} "
                f"--dedicated-l1 {l1_splits[i][0]} {l1_splits[i][1]}"
            )

            err = os.system(command)
            if err != 0:
                print(f"[DynMultiDeeployTestRunner] Error during the MultiDeeploy execution for models {models_name[0]}_{g0} and {models_name[1]}_{g1}")
                exit(1)

        # Network.h code fusion
        network_h_path = f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/{models_name[0]}_x_{models_name[1]}/Network.h"
        command = f"cp /app/Deeploy/DynMultiDeeploy/Templates/Network.h {network_h_path}"
        err = os.system(command)
        if err != 0:
            print(f"[DynMultiDeeployTestRunner] Error during the creation of Network.h for models {models_name[0]} and {models_name[1]}")
            exit(1)
        fused_variables = fuse_variables(
            files=[
                Path(f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/{models_name[0]}_x_{models_name[1]}/{models_name[0]}_2_x_{models_name[1]}_6/Network.h"),
                Path(f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/{models_name[0]}_x_{models_name[1]}/{models_name[0]}_4_x_{models_name[1]}_4/Network.h"),
                Path(f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/{models_name[0]}_x_{models_name[1]}/{models_name[0]}_6_x_{models_name[1]}_2/Network.h")
            ]
        )
        with open(network_h_path, "a") as f:
            f.write(fused_variables)
            f.write("#endif\n")

        # testinputs/testoutputs copying and renaming
        testinout_h_path = f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/{models_name[0]}_x_{models_name[1]}/"
        command = f"cp {testinout_h_path}/{models_name[0]}_2_x_{models_name[1]}_6/testinputs.h {testinout_h_path}/testinputs.h"
        err = os.system(command)
        if err != 0:
            print(f"[DynMultiDeeployTestRunner] Error during the copying of testinputs.h for models {models_name[0]} and {models_name[1]}")
            exit(1)
        command = f"cp {testinout_h_path}/{models_name[0]}_2_x_{models_name[1]}_6/testoutputs.h {testinout_h_path}/testoutputs.h"
        err = os.system(command)
        if err != 0:
            print(f"[DynMultiDeeployTestRunner] Error during the copying of testoutputs.h for models {models_name[0]} and {models_name[1]}")
            exit(1)

        testinputs_text = Path(testinout_h_path + "testinputs.h").read_text()
        testinputs_text = strip_core_suffix(testinputs_text, models_name)
        with open(testinout_h_path + "testinputs.h", "w") as f:
            f.write(testinputs_text)

        testoutputs_text = Path(testinout_h_path + "testoutputs.h").read_text()
        testoutputs_text = strip_core_suffix(testoutputs_text, models_name)
        # Also strip the core-count suffix from the uppercase #define macros (e.g. MINIMNV2_2_OUTPUTTYPE)
        testoutputs_text = strip_core_suffix(testoutputs_text, [model_name.upper() for model_name in models_name])
        with open(testinout_h_path + "testoutputs.h", "w") as f:
            f.write(testoutputs_text)

        # Network.c code fusion
        network_c_path = f"{test_c_path}/Network.c"
        fused_network_c = fuse_network_c(
            template_path=Path("/app/Deeploy/DynMultiDeeploy/Templates/Network.c"),
            files=[
                Path(f"{test_c_path}/{models_name[0]}_{g0}_x_{models_name[1]}_{g1}/Network.c")
                for g0, g1 in cores_splits
            ],
            core_splits=[(g0, g1) for g0, g1 in cores_splits],
            model_names=models_name
        )
        with open(network_c_path, "w") as f:
            f.write(fused_network_c)

        print(f"[DynMultiDeeployTestRunner] Successfully generated the dynamic-reconfiguration test for models {models_name[0]} and {models_name[1]} at {test_c_path}")

    print(f"[DynMultiDeeployTestRunner] Starting the simulation of the dynamic-reconfiguration test for models {models_name[0]} and {models_name[1]}")
    # Main code renaming
    update_dynmultideeploy_test_main(models_name)

    # Run the fused dynamic-reconfiguration test
    test_name = test_c_path.split("/")[-1]

    # CMake configuration for the dynamic-reconfiguration test
    # Uses its own build directory (distinct from TEST_SIRACUSA/build, used by the
    # per-configuration sub-builds above) since ENABLE_DYN_RECONFIG_TEST is a cached
    # CMake option: reusing the same build directory would let it leak between runs
    # and make later sub-builds wrongly pick up Platforms/DynSiracusa.
    dyn_build_dir = "TEST_SIRACUSA/build_dyn"
    command = (
        "$CMAKE -D TOOLCHAIN=LLVM -D TOOLCHAIN_INSTALL_DIR=/app/install/llvm "
        f"-D GENERATED_SOURCE={test_c_path} "
        "-D platform=Siracusa -D NUM_CORES=8 -D banshee_simulation=OFF -D gvsoc_simulation=ON "
        f"-B {dyn_build_dir} -DENABLE_DYNMULTIDEEPLOY_TEST=ON -DENABLE_DYN_RECONFIG_TEST=ON "
        f"-D TESTNAME={test_name} .."
    )
    print(f"[DynMultiDeeploy] CMake configuration command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[DynMultiDeeployTestRunner] Error during the cmake configuration of dynamic-reconfiguration test {test_name}")
        exit(1)

    # Build the dynamic-reconfiguration test
    command = f"$CMAKE --build /app/Deeploy/DynMultiDeeployTest/{dyn_build_dir} --target " + test_name
    print(f"[DynMultiDeeploy] Build command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[DynMultiDeeployTestRunner] Error during the build of dynamic-reconfiguration test {test_name}")
        exit(1)

    # Simulation of the dynamic-reconfiguration test
    command = f"$CMAKE --build {dyn_build_dir} --target gvsoc_" + test_name
    print(f"[DynMultiDeeploy] Simulation command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[DynMultiDeeployTestRunner] Error during the simulation of dynamic-reconfiguration test {test_name}")
        exit(1)
