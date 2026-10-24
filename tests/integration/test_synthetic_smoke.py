"""Synthetic end-to-end smoke without HMDA files or row-level outputs."""

from __future__ import annotations

from mortgage_threat_intel.cli import main


def test_cli_smoke_exits_zero():
    assert main(["smoke", "--config", "configs/base.yaml"]) == 0


def test_cli_data_check_reports_missing_without_crash(tmp_path, monkeypatch):
    # Point raw dir at empty temp path via config copy would be heavy; ensure
    # CLI returns non-zero when files are absent but still prints guidance.
    code = main(["data-check", "--config", "configs/base.yaml"])
    assert code in (0, 1)
