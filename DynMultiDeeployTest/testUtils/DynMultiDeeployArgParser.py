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

    args = parser.parse_args()
    return args