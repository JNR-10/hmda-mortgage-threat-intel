"""DP accountant metadata and achieved ε/δ serialization."""

from __future__ import annotations

from mortgage_threat_intel.privacy.accounting import (
    extract_accountant_metadata,
    serialize_privacy_metadata,
)


def test_serialize_keeps_target_and_achieved_distinct():
    meta = extract_accountant_metadata(
        target_epsilon=4.0,
        achieved_epsilon=3.7,
        delta=1e-5,
        sample_rate=0.1,
        steps=100,
        noise_multiplier=1.2,
        max_grad_norm=1.0,
        n_train=1000,
        adjacency="record_level",
        device="cpu",
        accountant_name="RDPAccountant",
    )
    payload = serialize_privacy_metadata(meta)
    assert payload["target_epsilon"] == 4.0
    assert payload["achieved_epsilon"] == 3.7
    assert payload["delta"] == 1e-5
    assert payload["adjacency"] == "record_level"
    assert "do not report target_epsilon" in payload["note"]
