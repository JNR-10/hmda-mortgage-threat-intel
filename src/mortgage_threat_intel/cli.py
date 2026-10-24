"""CLI entry point — dispatch only; no accidental full experiments on import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mortgage_threat_intel import __version__
from mortgage_threat_intel.config import default_config_dir, load_base_config


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"mortgage-threat-intel {__version__}")
    return 0


def _cmd_data_check(args: argparse.Namespace) -> int:
    from mortgage_threat_intel.data.ingest import data_check, format_data_check_report

    config = load_base_config(Path(args.config))
    report = data_check(config)
    print(format_data_check_report(report))
    return 0 if report.ok else 1


def _cmd_profile(args: argparse.Namespace) -> int:
    from mortgage_threat_intel.data.profile import write_data_profile

    config = load_base_config(Path(args.config))
    out = Path(args.output)
    try:
        profile = write_data_profile(config, out)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote aggregate profile to {out}")
    print(f"development_years={len(profile.get('development_profiles', []))}")
    print(f"test_year_meta={profile.get('test_year_meta')}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Synthetic smoke: labels, splits, baselines, privacy aggregates — no HMDA."""
    import pandas as pd

    from mortgage_threat_intel.config import TargetConfig
    from mortgage_threat_intel.data.labels import map_action_taken
    from mortgage_threat_intel.data.splits import (
        assert_no_test_year_in_fit,
        partition_mia_members,
        split_by_year,
    )
    from mortgage_threat_intel.experiments.baseline import run_baselines
    from mortgage_threat_intel.experiments.privacy import run_privacy
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

    print("synthetic-smoke OK (data contracts)")
    print(f"  rows_labeled={len(labeled.frame)} pos={labeled.n_positive} neg={labeled.n_negative}")
    print(f"  excluded={labeled.n_excluded} exclusion_counts={labeled.exclusion_counts}")
    print(f"  train={len(split.train)} val={len(split.validation)}")
    print(f"  mia_members={len(mia.members)} mia_non_members={len(mia.non_members)}")
    print(f"  target_config={TargetConfig().label_name}")

    if args.with_models:
        base_out = run_baselines(config=config, use_synthetic=True)
        priv_out = run_privacy(base_config=config, use_synthetic=True)
        print("synthetic-smoke OK (baselines + privacy aggregates)")
        print(f"  baseline_run={base_out['manifest']['run_id']}")
        print(f"  privacy_run={priv_out['manifest']['run_id']}")
        print(
            "  achieved_eps="
            f"{priv_out['dp_privacy_metadata']['achieved_epsilon']} "
            f"delta={priv_out['dp_privacy_metadata']['delta']}"
        )
    print("  note: no row-level artifacts written")
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    from mortgage_threat_intel.experiments.baseline import run_baselines

    config = load_base_config(Path(args.config))
    payload = run_baselines(config=config, use_synthetic=args.synthetic)
    print(f"baseline OK run_id={payload['manifest']['run_id']}")
    print(payload["metrics"])
    return 0


def _cmd_privacy(args: argparse.Namespace) -> int:
    from mortgage_threat_intel.config import load_privacy_config
    from mortgage_threat_intel.experiments.privacy import run_privacy

    base = load_base_config(Path(args.config))
    privacy = load_privacy_config(Path(args.privacy_config))
    payload = run_privacy(base_config=base, privacy_config=privacy, use_synthetic=args.synthetic)
    print(f"privacy OK run_id={payload['manifest']['run_id']}")
    print(f"dp_metadata={payload['dp_privacy_metadata']}")
    print(f"mia={payload['mia']}")
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
    p_data.add_argument("--config", default=str(default_config_dir() / "base.yaml"))
    p_data.set_defaults(func=_cmd_data_check)

    p_profile = sub.add_parser("profile", help="Write aggregate 2023/2024 profile JSON")
    p_profile.add_argument("--config", default=str(default_config_dir() / "base.yaml"))
    p_profile.add_argument(
        "--output",
        default="outputs/data_profile.json",
        help="Aggregate profile output path",
    )
    p_profile.set_defaults(func=_cmd_profile)

    p_smoke = sub.add_parser("smoke", help="Synthetic end-to-end smoke without HMDA files")
    p_smoke.add_argument("--config", default=str(default_config_dir() / "base.yaml"))
    p_smoke.add_argument(
        "--with-models",
        action="store_true",
        help="Also run synthetic baselines + privacy (slower)",
    )
    p_smoke.set_defaults(func=_cmd_smoke)

    p_base = sub.add_parser("baseline", help="Train LR/HGB baselines (aggregate metrics)")
    p_base.add_argument("--config", default=str(default_config_dir() / "base.yaml"))
    p_base.add_argument(
        "--synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use synthetic fixture (default true; required without HMDA)",
    )
    p_base.set_defaults(func=_cmd_baseline)

    p_priv = sub.add_parser("privacy", help="Matched MLP DP-SGD + confidence MIA")
    p_priv.add_argument("--config", default=str(default_config_dir() / "base.yaml"))
    p_priv.add_argument(
        "--privacy-config",
        default=str(default_config_dir() / "privacy.yaml"),
    )
    p_priv.add_argument(
        "--synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_priv.set_defaults(func=_cmd_privacy)

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
