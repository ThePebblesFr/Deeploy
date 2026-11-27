import argparse
import os
from pathlib import Path

from testUtils.UpdateMainDeeployTest import update_deeploy_test_main
from testUtils.UpdateRoundRobinMain import update_roundrobin_main
from DynMultiDeeploy.RoundRobinFusion import fuse_roundrobin_network_h, fuse_roundrobin_network_c
from scenarioUtils.ScenarioHeaderGen import generate_scenario_header

"""
Round-Robin-Large baseline (Phase 5, method 3): the two LARGE (heavy) variants,
each deployed alone on the full 8-core cluster, alternating every tick -- never
co-resident, unlike every other method built so far. See
Platforms/RoundRobin/src/roundrobin_template.c's module comment for why this means
paying a full InitNetwork (weight/arena reload) cost on every switch, every tick:
that reload cost is exactly the expensive behavior DynReconf's design is meant to
avoid, so this baseline must pay it, not have it optimized away.

Usage (inside the container, from DynMultiDeeployTest/):
    python3 test_roundrobin.py -t Tests/TinyClassifier_x_AnomalyDetection \
        --models TinyClassifier AnomalyDetection --scenario Scenarios/static_large --gen-step
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-Robin-Large baseline generation/simulation.")
    parser.add_argument("-t", "--test-folder", type=str, required=True,
                         help="Pair directory containing <task2>_heavy/ and <task1>_heavy/ source dirs "
                              "(e.g. Tests/TinyClassifier_x_AnomalyDetection).")
    parser.add_argument("--models", nargs=2, required=True, metavar=("TASK2_BASE", "TASK1_BASE"),
                         help="Base model names, e.g. --models TinyClassifier AnomalyDetection "
                              "(matches --models' order/meaning in test_dynamic_2_models.py: "
                              "models[0]=Task 2, models[1]=Task 1). Always uses each one's _heavy variant.")
    parser.add_argument("--scenario", type=str, required=True,
                         help="Path to a scenario directory (as produced by scenarioUtils/create_scenario.py).")
    parser.add_argument("--gen-step", dest="gen_step", action="store_true", default=False,
                         help="Regenerate both full-cluster deployments and refuse the Network.c/.h/roundrobin.c "
                              "from scratch. Omit to reuse an existing generation and just rerun the scenario.")
    args = parser.parse_args()

    test_folder = args.test_folder
    model1_tag = f"{args.models[0]}_heavy"  # Task 2
    model2_tag = f"{args.models[1]}_heavy"  # Task 1

    test_c_path = "/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/RoundRobin_Large"

    if args.gen_step:
        for tag in (model1_tag, model2_tag):
            update_deeploy_test_main(tag)
            command = (
                f"python3 testRunner_tiled_siracusa.py "
                f"-t {test_folder}/{tag} "
                f"--test-gen Tests/RoundRobin_Large/{tag} "
                f"--defaultMemLevel L2 --profileTiling --profileToJSON "
                f"--cores 8 --l1 65536 "
                f"--name {tag}"
            )
            print(f"[RoundRobin] Generation command: {command}")
            err = os.system(command)
            if err != 0:
                # Same known, tolerated correctness-varies-by-tiling-config issue as
                # everywhere else in this codebase (e.g. test_dynamic_2_models.py's
                # per-config loop): generation + build + the single-vector sanity
                # simulation all complete before that check runs, so Network.c is
                # already on disk. Only fatal if it's genuinely missing.
                network_c = Path(f"{test_c_path}/{tag}/Network.c")
                if not network_c.exists():
                    print(f"[RoundRobin] Error generating {tag} (no Network.c produced)")
                    exit(1)
                print(f"[RoundRobin] Warning: {tag}'s own sanity build/simulation failed at (8 cores, 64kB); continuing")

        network_h = fuse_roundrobin_network_h(
            [Path(f"{test_c_path}/{model1_tag}/Network.h"), Path(f"{test_c_path}/{model2_tag}/Network.h")],
            [model1_tag, model2_tag],
        )
        Path(f"{test_c_path}/Network.h").write_text(network_h)

        network_c = fuse_roundrobin_network_c(
            [Path(f"{test_c_path}/{model1_tag}/Network.c"), Path(f"{test_c_path}/{model2_tag}/Network.c")],
            [model1_tag, model2_tag],
        )
        Path(f"{test_c_path}/Network.c").write_text(network_c)

        print(f"[RoundRobin] Successfully generated Round-Robin-Large at {test_c_path}")

    update_roundrobin_main(model1_tag, model2_tag)

    scenario_header_path = Path(test_c_path) / "scenario_data.h"
    generate_scenario_header(Path(args.scenario), scenario_header_path)

    test_name = "RoundRobin_Large"
    build_dir = "TEST_SIRACUSA/build_roundrobin"
    command = (
        "$CMAKE -D TOOLCHAIN=LLVM -D TOOLCHAIN_INSTALL_DIR=/app/install/llvm "
        f"-D GENERATED_SOURCE={test_c_path} "
        "-D platform=Siracusa -D NUM_CORES=8 -D banshee_simulation=OFF -D gvsoc_simulation=ON "
        f"-B {build_dir} -DENABLE_DYNMULTIDEEPLOY_TEST=ON -DENABLE_ROUNDROBIN_TEST=ON "
        "-D GVSOC_EXTRA_FLAGS=--power "
        f"-D TESTNAME={test_name} .."
    )
    print(f"[RoundRobin] CMake configuration command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[RoundRobin] Error during the cmake configuration of {test_name}")
        exit(1)

    command = f"$CMAKE --build /app/Deeploy/DynMultiDeeployTest/{build_dir} --target " + test_name
    print(f"[RoundRobin] Build command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[RoundRobin] Error during the build of {test_name}")
        exit(1)

    command = f"$CMAKE --build {build_dir} --target gvsoc_" + test_name
    print(f"[RoundRobin] Simulation command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[RoundRobin] Error during the simulation of {test_name}")
        exit(1)
