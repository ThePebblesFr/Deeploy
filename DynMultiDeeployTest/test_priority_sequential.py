import argparse
import os
from pathlib import Path

from testUtils.UpdateRoundRobinMain import update_priority_sequential_main
from scenarioUtils.ScenarioHeaderGen import generate_scenario_header

"""
Priority-Sequential baseline (Phase 5, method 4): reuses Round-Robin-Large's exact
already-generated Network.c/.h (same two full-cluster heavy deployments, same
InitNetwork_*/RunNetwork_* symbols -- run test_roundrobin.py --gen-step first if
these don't exist yet). Only the main-loop dispatch logic differs: true priority
preemption -- each tick, only the higher-priority (more critical) task actually
runs; the other is skipped outright, not deferred. See
Platforms/PrioritySequential/src/priority_sequential_template.c's module comment
for the full priority/staleness/theta-attribution rationale.

Usage (inside the container, from DynMultiDeeployTest/, after test_roundrobin.py
--gen-step has produced TEST_SIRACUSA/Tests/RoundRobin_Large/Network.c/.h):
    python3 test_priority_sequential.py --models TinyClassifier AnomalyDetection \
        --scenario Scenarios/static_large
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Priority-Sequential baseline simulation.")
    parser.add_argument("--models", nargs=2, required=True, metavar=("TASK2_BASE", "TASK1_BASE"),
                         help="Same meaning as test_roundrobin.py's --models. Must match whichever pair "
                              "test_roundrobin.py --gen-step already generated into TEST_SIRACUSA/Tests/"
                              "RoundRobin_Large -- this script does not regenerate models.")
    parser.add_argument("--scenario", type=str, required=True,
                         help="Path to a scenario directory (as produced by scenarioUtils/create_scenario.py).")
    args = parser.parse_args()

    model1_tag = f"{args.models[0]}_heavy"  # Task 2
    model2_tag = f"{args.models[1]}_heavy"  # Task 1

    test_c_path = "/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/RoundRobin_Large"
    if not Path(f"{test_c_path}/Network.c").exists():
        print(f"[PrioritySequential] {test_c_path}/Network.c not found -- run "
              f"test_roundrobin.py --gen-step first (Priority-Sequential reuses its fused Network.c/.h).")
        exit(1)

    update_priority_sequential_main(model1_tag, model2_tag)

    scenario_header_path = Path(test_c_path) / "scenario_data.h"
    generate_scenario_header(Path(args.scenario), scenario_header_path)

    test_name = "PrioritySequential"
    build_dir = "TEST_SIRACUSA/build_priority_sequential"
    command = (
        "$CMAKE -D TOOLCHAIN=LLVM -D TOOLCHAIN_INSTALL_DIR=/app/install/llvm "
        f"-D GENERATED_SOURCE={test_c_path} "
        "-D platform=Siracusa -D NUM_CORES=8 -D banshee_simulation=OFF -D gvsoc_simulation=ON "
        f"-B {build_dir} -DENABLE_DYNMULTIDEEPLOY_TEST=ON -DENABLE_PRIORITY_SEQUENTIAL_TEST=ON "
        "-D GVSOC_EXTRA_FLAGS=--power "
        f"-D TESTNAME={test_name} .."
    )
    print(f"[PrioritySequential] CMake configuration command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[PrioritySequential] Error during the cmake configuration of {test_name}")
        exit(1)

    command = f"$CMAKE --build /app/Deeploy/DynMultiDeeployTest/{build_dir} --target " + test_name
    print(f"[PrioritySequential] Build command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[PrioritySequential] Error during the build of {test_name}")
        exit(1)

    command = f"$CMAKE --build {build_dir} --target gvsoc_" + test_name
    print(f"[PrioritySequential] Simulation command: {command}")
    err = os.system(command)
    if err != 0:
        print(f"[PrioritySequential] Error during the simulation of {test_name}")
        exit(1)
