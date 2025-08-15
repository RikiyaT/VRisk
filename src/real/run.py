#!/usr/bin/env python3
"""
run.py  –  generic entry point for diversification sweeps.

Usage
-----
python run.py <param> [metric] [-c CONFIG] [-o OUTDIR] [-d DATASET]

Examples
--------
python run.py beta                 # metric from YAML
python run.py k dcg -c my.yaml      # override metric with "dcg"
python run.py beta -d MOVIELENS     # override dataset with "MOVIELENS"
"""
import argparse
import os
import sys
import yaml

from utils.experiment import run_experiment



def load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"[config] file not found: {path}")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)



def main() -> None:
    parser = argparse.ArgumentParser(description="Diversification sweeps")
    parser.add_argument("param", help="hyper-parameter key to sweep")
    parser.add_argument(
        "metric",
        nargs="?",
        help="(optional) metric override: rel | dcg | map | mrr",
    )
    parser.add_argument(
        "-c", "--config", default="config/config.yaml", help="YAML path"
    )
    parser.add_argument(
        "-d", "--dataset", help="dataset override (if not specified, uses config dataset)"
    )
    parser.add_argument("--dry", action="store_true", help="print only")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # ---- dataset override -------------------------------------------
    if args.dataset:
        cfg['dataset'] = args.dataset

    # ---- sweep values ----------------------------------------------
    sweep_vals = cfg["experiment_params"].get(args.param)
    if sweep_vals is None:
        sys.exit(f"[config] no values for '{args.param}' in experiment_params")

    # ---- metric override -------------------------------------------
    if args.metric:
        cfg["metric"] = args.metric.lower()

    methods = cfg.get("methods", [])
    if not methods:
        sys.exit("[config] list at least one method under 'methods'")

    outdir = f"../../results/real/{cfg['dataset']}/{args.param}"

    # ---- summary ---------------------------------------------------
    print("┌─ experiment")
    print(f"│ dataset     : {cfg['dataset']}")
    print(f"│ metric      : {cfg['metric']}")
    print(f"│ sweep param : {args.param}")
    print(f"│ sweep vals  : {sweep_vals}")
    print(f"│ methods     : {methods}")
    print(f"│ output dir  : {outdir}")
    print("└──────────────────────────")
    if args.dry:
        print("[dry-run] finished.")
        return

    os.makedirs(outdir, exist_ok=True)

    run_experiment(
        sweep_key=args.param,
        sweep_values=sweep_vals,
        methods_cfg=methods,
        global_cfg=cfg,      # includes metric (possibly overridden)
        out_dir=outdir,
    )


if __name__ == "__main__":
    main()
