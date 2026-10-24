"""CLI entry point — dispatch only; no accidental full experiments on import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mortgage_threat_intel import __version__
from mortgage_threat_intel.config import default_config_dir, load_base_config
from mortgage_threat_intel.data.ingest import data_check, format_data_check_report


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"mortgage-threat-intel {__version__}")
    return 0


def _cmd_data_check(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_base_config(config_path)
    report = data_check(config)
    print(format_data_check_report(report))
    return 0 if report.ok else 1


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Synthetic smoke: load config, map labels, split years — no HMDA files."""
    import pandas as pd

    from mortgage_threat_intel.config import TargetConfig
    from mortgage_threat_intel.data.labels import map_action_taken
    from mortgage_threat_intel.data.splits import (
        assert_no_test_year_in_fit,
        partition_mia_members,
        split_by_year,
    )
    from mortgage_threat_intel.seeding import seed_everything

    config = load_base_config(Path(args.config))
    seed_everything(config.seed)

    fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "synthetic_hmda.csv"
    if not fixture.is_file():
        print(f"Missing synthetic fixture: {fixture}", file=sys.stderr)
        return 1

    df = pd.read_csv(fixture)
    labeled = map_action_taken(df, config.target)
    split = split_by_year(labeled.frame, config, include_test=False)
    assert_no_test_year_in_fit([split.train, split.validation], test_year=config.years.test)
    mia = partition_mia_members(split.train, config)

    print("synthetic-smoke OK")
    print(f"  rows_labeled={len(labeled.frame)} pos={labeled.n_positive} neg={labeled.n_negative}")
    print(f"  excluded={labeled.n_excluded} exclusion_counts={labeled.exclusion_counts}")
    print(f"  train={len(split.train)} val={len(split.validation)}")
    print(f"  mia_members={len(mia.members)} mia_non_members={len(mia.non_members)}")
    print(f"  target_config={TargetConfig().label_name}")
    print("  note: no row-level artifacts written")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mortgage-threat-intel",
        description=(
            "Offline HMDA proxy research pipeline. "
            "Does not evaluate Fannie Mae Desktop Underwriter."
        ),
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    sub = parser.add_subparsers(dest="command")

    p_data = sub.add_parser("data-check", help="Validate local HMDA files (no download)")
    p_data.add_argument(
        "--config",
        default=str(default_config_dir() / "base.yaml"),
        help="Path to base config YAML",
    )
    p_data.set_defaults(func=_cmd_data_check)

    p_smoke = sub.add_parser("smoke", help="Synthetic end-to-end smoke without HMDA files")
    p_smoke.add_argument(
        "--config",
        default=str(default_config_dir() / "base.yaml"),
        help="Path to base config YAML",
    )
    p_smoke.set_defaults(func=_cmd_smoke)

    p_ver = sub.add_parser("version", help="Print package version")
    p_ver.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False) and not args.command:
        return _cmd_version(args)
    if not args.command:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
