import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Dynamic Multi-Deeploy workload scheduling over multiple models."
    )

    parser.add_argument(
        "-t", "--test-folder",
        type=str,
        required=True,
        help="Folder that contains the multi-model test directory."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to run (e.g., --models modelA modelB)"
    )

    parser.add_argument(
        '--gen-step',
        dest = 'gen_step',
        action = 'store_true',
        default = False,
        help = 'Skip network generation\n'
    )

    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Path to a JSON file listing the 4 per-mode configs (variant0/variant1/cores0/cores1/"
             "l1_0/l1_1/v0/v1), shaped like Configs/tinyclassifier_anomalydetection_4mode.json. "
             "Defaults to that file (today's TinyClassifier/AnomalyDetection 4-mode setup) if omitted. "
             "Order must match dynmultideeploytest_template.c's 4 hardcoded case blocks "
             "(LOW_ENERGY, CRITICAL_TASK1, CRITICAL_TASK2, BOTH_CRITICAL)."
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Path to a scenario directory (as produced by scenarioUtils/create_scenario.py) "
             "containing scenario_config.json/task1_inputs.npz/task2_inputs.npz. When given, "
             "generates scenario_data.h and copies it into the project before the CMake build, "
             "driving the context-based FSM instead of the plain --gen-step demo."
    )

    args = parser.parse_args()
    return args