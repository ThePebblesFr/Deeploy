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

    parser.add_argument(
        "--doublebuffer",
        nargs="+",
        type=int,
        default=None,
        help="Per-model 0/1 flags selecting Deeploy's --doublebuffer tiling for that "
             "model's own generation (e.g., --doublebuffer 0 1 to double-buffer only "
             "the second model). Defaults to single-buffered (0) for every model when "
             "omitted. MultiDeeploy's fusion detects each layer's actual buffering "
             "shape from the generated source regardless of this flag, so this only "
             "controls what Deeploy itself generates, not how the fusion re-emits it."
    )

    args = parser.parse_args()

    if args.doublebuffer is None:
        args.doublebuffer = [0] * len(args.models)

    # Sanity check: all lists must have the same length
    if not (len(args.models) == len(args.dedicated_cores) == len(args.dedicated_l1) == len(args.doublebuffer)):
        raise ValueError(
            "Error: --models, --dedicated-cores, --dedicated-l1, and --doublebuffer "
            "(when given) must have the same number of elements!"
        )

    return args