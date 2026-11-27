import os
import sys
import re
from pathlib import Path
import json

from testUtils.DynMultiDeeployArgParser import parse_arguments
from testUtils.UpdateMainDeeployTest import update_dynmultideeploy_test_main
from testUtils.ConfigLoader import load_configs, DEFAULT_CONFIGS_PATH
from DynMultiDeeploy.HeaderFusion import fuse_variables, strip_variant_core_suffix
from DynMultiDeeploy.SourceFusion import fuse_network_c
from scenarioUtils.ScenarioHeaderGen import generate_scenario_header

if __name__ == "__main__":

    args = parse_arguments()
    test_folder = args.test_folder
    models_name = args.models

    test_c_path = "/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/" + models_name[0] + "_x_" + models_name[1]

    # One entry per operating mode: LE, CRITICAL_TASK1, CRITICAL_TASK2, BOTH_CRITICAL,
    # in that order -- loaded from --configs (or Configs/tinyclassifier_
    # anomalydetection_4mode.json by default) rather than hardcoded here, so a
    # different model pair/core-split/L1 choice no longer requires editing this
    # script. Model_1 = models_name[0] (Task 2, e.g. TinyClassifier), Model_2 =
    # models_name[1] (Task 1, e.g. AnomalyDetection) -- matches the --models CLI
    # order and the Model_i placeholder substitution in dynmultideeploytest_template.c.
    # "variant0"/"variant1" are the real per-task model directories (light/heavy),
    # e.g. "TinyClassifier_light" vs "TinyClassifier_heavy" -- swapping in a config
    # with genuinely different model directories still requires updating
    # dynmultideeploytest_template.c's literal per-mode symbol names to match (that
    # template isn't generated from this config list yet -- Phase 3).
    configs_path = Path(args.configs) if args.configs else DEFAULT_CONFIGS_PATH
    CONFIGS = load_configs(configs_path)
    print(f"[DynMultiDeeployTestRunner] Loaded {len(CONFIGS)} mode configs from {configs_path}")

    # Every distinct (task, variant) directory used anywhere in CONFIGS -- used to
    # dedupe PI_L2 weight variables per variant (light shared across the configs that
    # reuse it, heavy kept separate) rather than per base task name.
    VARIANT_TAGS = sorted({cfg["variant0"] for cfg in CONFIGS} | {cfg["variant1"] for cfg in CONFIGS})

    if args.gen_step:

        # MultiDeeployTests generation

        for cfg in CONFIGS:
            model0 = f"{cfg['variant0']}_{cfg['cores0']}"
            model1 = f"{cfg['variant1']}_{cfg['cores1']}"

            # Create a folder for each configuration
            os.system(f"mkdir -p {test_c_path}/{model0}_x_{model1}/{model0}")
            os.system(f"mkdir -p {test_c_path}/{model0}_x_{model1}/{model1}")

            # Run MultiDeeploy for each configuration. doublebuffer0/doublebuffer1
            # are optional per-mode fields (0/1, default 0/SB) -- added once
            # MultiDeeploy's fusion gained real double-buffered-closure support;
            # omitted entirely in a config file, or in older ones predating this,
            # they default to SB for both models, matching pre-DB-support behavior.
            db0 = cfg.get("doublebuffer0", 0)
            db1 = cfg.get("doublebuffer1", 0)
            command = (
                f"python3 testRunner_2_models.py "
                f"-t {args.test_folder} "
                f"--models {model0} {model1} "
                f"--dedicated-cores {cfg['cores0']} {cfg['cores1']} "
                f"--dedicated-l1 {cfg['l1_0']} {cfg['l1_1']} "
                f"--doublebuffer {db0} {db1}"
            )

            err = os.system(command)
            if err != 0:
                # testRunner_2_models.py's own tail end does a full plain-MultiDeeployTest
                # sanity build+GVSoC-run of this single config, on top of producing the
                # fused Network.c we actually need. That sanity run can itself crash
                # GVSoC (observed: a pathological WorkloadScheduler round split when one
                # task's per-layer time massively dwarfs the other's) -- unrelated to
                # whether the Network.c this config needs to contribute was produced.
                # Only treat this as fatal if that file is actually missing.
                config_network_c = Path(f"{test_c_path}/{model0}_x_{model1}/Network.c")
                if not config_network_c.exists():
                    print(f"[DynMultiDeeployTestRunner] Error during the MultiDeeploy execution for models {model0} and {model1} (no Network.c produced)")
                    exit(1)
                print(f"[DynMultiDeeployTestRunner] Warning: config {model0}_x_{model1}'s own sanity build/simulation failed, "
                      f"but its Network.c was produced; continuing")

        # Network.h code fusion
        network_h_path = f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/{models_name[0]}_x_{models_name[1]}/Network.h"
        command = f"cp /app/Deeploy/DynMultiDeeploy/Templates/Network.h {network_h_path}"
        err = os.system(command)
        if err != 0:
            print(f"[DynMultiDeeployTestRunner] Error during the creation of Network.h for models {models_name[0]} and {models_name[1]}")
            exit(1)
        fused_variables = fuse_variables(
            files=[
                Path(
                    f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/{models_name[0]}_x_{models_name[1]}/"
                    f"{cfg['variant0']}_{cfg['cores0']}_x_{cfg['variant1']}_{cfg['cores1']}/Network.h"
                )
                for cfg in CONFIGS
            ]
        )
        with open(network_h_path, "a") as f:
            f.write(fused_variables)
            f.write("#endif\n")

        # testinputs/testoutputs copying and renaming (any one config's testinputs/
        # testoutputs works: the test vector is the same regardless of core/L1 split)
        testinout_h_path = f"/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/Tests/{models_name[0]}_x_{models_name[1]}/"
        source_cfg = CONFIGS[0]
        source_config_dir = f"{source_cfg['variant0']}_{source_cfg['cores0']}_x_{source_cfg['variant1']}_{source_cfg['cores1']}"
        command = f"cp {testinout_h_path}/{source_config_dir}/testinputs.h {testinout_h_path}/testinputs.h"
        err = os.system(command)
        if err != 0:
            print(f"[DynMultiDeeployTestRunner] Error during the copying of testinputs.h for models {models_name[0]} and {models_name[1]}")
            exit(1)
        command = f"cp {testinout_h_path}/{source_config_dir}/testoutputs.h {testinout_h_path}/testoutputs.h"
        err = os.system(command)
        if err != 0:
            print(f"[DynMultiDeeployTestRunner] Error during the copying of testoutputs.h for models {models_name[0]} and {models_name[1]}")
            exit(1)

        testinputs_text = Path(testinout_h_path + "testinputs.h").read_text()
        testinputs_text = strip_variant_core_suffix(testinputs_text, VARIANT_TAGS)
        with open(testinout_h_path + "testinputs.h", "w") as f:
            f.write(testinputs_text)

        testoutputs_text = Path(testinout_h_path + "testoutputs.h").read_text()
        testoutputs_text = strip_variant_core_suffix(testoutputs_text, VARIANT_TAGS)
        # Also strip the core-count suffix from the uppercase #define macros (e.g. ANOMALYDETECTION_LIGHT_4_OUTPUTTYPE)
        testoutputs_text = strip_variant_core_suffix(testoutputs_text, [tag.upper() for tag in VARIANT_TAGS])
        with open(testinout_h_path + "testoutputs.h", "w") as f:
            f.write(testoutputs_text)

        # Network.c code fusion
        network_c_path = f"{test_c_path}/Network.c"
        fused_network_c = fuse_network_c(
            template_path=Path("/app/Deeploy/DynMultiDeeploy/Templates/Network.c"),
            files=[
                Path(f"{test_c_path}/{cfg['variant0']}_{cfg['cores0']}_x_{cfg['variant1']}_{cfg['cores1']}/Network.c")
                for cfg in CONFIGS
            ],
            core_splits=[(cfg["cores0"], cfg["cores1"]) for cfg in CONFIGS],
            # Real variant axis now: (v0, v1) = ("s"/"l" for TinyClassifier, "s"/"l"
            # for AnomalyDetection) per config, giving each of the 4 modes a distinct
            # RunNetworks/InitNetworks entry point even where core splits repeat (LE
            # and BOTH_CRITICAL both 4/4). Weight dedup keys off VARIANT_TAGS (the
            # actual light/heavy directory names): configs reusing the same variant
            # (e.g. LE and CRITICAL_TASK1 both using AnomalyDetection_light) collapse
            # to one shared PI_L2 weight copy; different variants of the same task
            # never do.
            variant_names=[(cfg["v0"], cfg["v1"]) for cfg in CONFIGS],
            variant_tags=VARIANT_TAGS
        )
        with open(network_c_path, "w") as f:
            f.write(fused_network_c)

        print(f"[DynMultiDeeployTestRunner] Successfully generated the dynamic-reconfiguration test for models {models_name[0]} and {models_name[1]} at {test_c_path}")

    print(f"[DynMultiDeeployTestRunner] Starting the simulation of the dynamic-reconfiguration test for models {models_name[0]} and {models_name[1]}")
    # Main code renaming + generation of the per-mode switch bodies from CONFIGS
    update_dynmultideeploy_test_main(models_name, CONFIGS)

    if args.scenario:
        # Generates scenario_data.h into test_c_path, alongside the fused Network.h --
        # both are on the "network" CMake target's include_directories(${GENERATED_SOURCE})
        # path, so `#include "scenario_data.h"` in dynmultideeploytest.c resolves the
        # same way `#include "Network.h"` already does.
        scenario_header_path = Path(test_c_path) / "scenario_data.h"
        generate_scenario_header(Path(args.scenario), scenario_header_path)

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
        # Empirically confirmed: GVSoC's power trigger (writes to 0x10000000 already
        # bracketing every RunNetworks_* call) emits one @power.measure_N@ line per
        # bracketed region, numbered sequentially -- this gives real per-inference phi
        # for free within a single continuous run.
        "-D GVSOC_EXTRA_FLAGS=--power "
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
