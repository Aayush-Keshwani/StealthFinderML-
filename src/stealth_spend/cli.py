"""
Command-line interface for Stealth-Spend.
Provides terminal commands to generate demo datasets, ingest raw CSV statements,
train all pipeline models, generate graphs, and export reports.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from stealth_spend.config import load_config
from stealth_spend.data_ingestion import SyntheticDataGenerator, normalize_to_canonical
from stealth_spend.pipeline import StealthSpendPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Stealth-Spend: Offline-first Financial Modeler & Anomaly Detector"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command 1: generate-demo
    sub_demo = subparsers.add_parser("generate-demo", help="Generate synthetic bank statement data")
    sub_demo.add_argument("--days", type=int, default=365, help="Number of calendar days to simulate")
    sub_demo.add_argument("--output", type=str, default="data/synthetic/synthetic_statement.csv", help="Output file path")

    # Command 2: ingest
    sub_ingest = subparsers.add_parser("ingest", help="Ingest and normalize raw statements")
    sub_ingest.add_argument("--input", type=str, required=True, help="Input CSV statement path")
    sub_ingest.add_argument("--output", type=str, default="data/interim/canonical_statement.csv", help="Output interim CSV path")

    # Command 3: run
    sub_run = subparsers.add_parser("run", help="Run full ML training and reporting pipeline")
    sub_run.add_argument("--config", type=str, default="config/settings.yml", help="Path to settings.yml")
    sub_run.add_argument("--input", type=str, default="transactions.csv", help="Input statement CSV path")
    sub_run.add_argument("--demo", action="store_true", help="Run with synthetic demo data")

    args = parser.parse_args()

    if args.command == "generate-demo":
        print(f"Generating synthetic statement with {args.days} days of data...")
        gen = SyntheticDataGenerator()
        df = gen.generate(days=args.days)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved synthetic dataset ({len(df)} transactions) to: {out_path}")

    elif args.command == "ingest":
        print(f"Ingesting raw statement from {args.input}...")
        df_raw = pd.read_csv(args.input)
        df_canonical = normalize_to_canonical(df_raw, source_filename=Path(args.input).name)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_canonical.to_csv(out_path, index=False)
        print(f"Normalized {len(df_canonical)} transactions into canonical schema at: {out_path}")

    elif args.command == "run":
        print("Starting Stealth-Spend pipeline execution...")
        pipeline = StealthSpendPipeline(config_path=args.config)
        pipeline.run(input_csv_path=args.input, use_demo=args.demo)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
