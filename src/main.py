#!/usr/bin/env python
"""FAERS Analytics Platform - Main Entry Point"""
import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_RESULTS_DIR = OUTPUTS_DIR / "eda_results"
MAPREDUCE_RESULTS_DIR = OUTPUTS_DIR / "mapreduce_results"


def clear_outputs():
    """Clear output directories."""
    for d in [EDA_RESULTS_DIR, MAPREDUCE_RESULTS_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    print("[OK] Outputs cleared")


def run_eda():
    """Run EDA pipeline."""
    print("\nRunning EDA...")
    eda_script = PROJECT_ROOT / "src" / "eda" / "00_eda_simple.py"
    if eda_script.exists():
        with open(eda_script) as f:
            exec(compile(f.read(), eda_script, 'exec'))
    print(f"[OK] EDA complete - {EDA_RESULTS_DIR}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    if not any([args.eda, args.all]):
        parser.print_help()
        return
    
    if args.all or args.eda:
        clear_outputs()
        run_eda()
    
    print("\n[OK] Pipeline completed!")


if __name__ == "__main__":
    main()