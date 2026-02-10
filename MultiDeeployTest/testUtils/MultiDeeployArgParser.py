import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Multi-Deeploy workload scheduling over multiple models."
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
        "--dedicated-cores",
        nargs="+",
        required=True,
        type=int,
        help="List of dedicated cores per model (e.g., --dedicated-cores 8 8)"
    )

    parser.add_argument(
        "--dedicated-l1",
        nargs="+",
        required=True,
        type=int,
        help="List of L1 memory sizes per model (e.g., --dedicated-l1 64000 64000)"
    )

    args = parser.parse_args()

    # Sanity check: all lists must have the same length
    if not (len(args.models) == len(args.dedicated_cores) == len(args.dedicated_l1)):
        raise ValueError(
            "Error: --models, --dedicated-cores, and --dedicated-l1 must have the same number of elements!"
        )
    
    return args