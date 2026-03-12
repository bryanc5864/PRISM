"""
PRISM command-line interface.

Usage:
    prism-run --system configs/skin.yaml --stage all
    prism-run --system configs/pancreas.yaml --stage resolve
"""

import argparse
import sys


def main():
    """CLI entry point that delegates to run_prism.py."""
    parser = argparse.ArgumentParser(
        description="PRISM: Progenitor Resolution via Invariance-Sensitive Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  prism-run --system configs/skin.yaml --stage all
  prism-run --system configs/pancreas.yaml --stage resolve
  prism-run --stage theory
        """,
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System config YAML (e.g., configs/skin.yaml). Default: skin.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "data", "train", "resolve", "trace",
                 "baselines", "ablation", "theory", "benchmark"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Training config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )
    args = parser.parse_args()

    # Build argv for run_prism.py
    run_args = ["--stage", args.stage, "--config", args.config, "--device", args.device]
    if args.system:
        run_args.extend(["--system", args.system])

    # Patch sys.argv and run
    sys.argv = ["run_prism.py"] + run_args

    # Import and run
    import importlib.util
    import os

    # Find run_prism.py relative to package
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_script = os.path.join(package_dir, "run_prism.py")

    if os.path.exists(run_script):
        spec = importlib.util.spec_from_file_location("run_prism", run_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    else:
        print(f"Error: run_prism.py not found at {run_script}")
        print("Run from the PRISM project root directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
