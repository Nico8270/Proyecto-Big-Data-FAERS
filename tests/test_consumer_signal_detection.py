"""
tests/test_consumer_signal_detection.py
=========================================
Unit test for FAERSConsumer.procesar_micro_batch().

Test scenario: a micro-batch of 77 records containing exactly 2 true signal pairs.

The test verifies:
  - Exactly 2 alerts are generated (no inflation).
  - The 2 alerts correspond to the correct (drug, pt) pairs.
  - No alert is generated for pairs below the minimum exposure threshold (N < 3).
  - No alert is generated for pairs that do not meet PRR > 2.0 AND p-value < 0.05.
  - Each (drug, pt) pair generates at most 1 alert (deduplication within batch).
  - alert counters in self.stats reflect the correct delta after the batch.

Run with:
    python -m pytest tests/test_consumer_signal_detection.py -v
"""

import sys
import queue
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_caso(pid: int, drugs: list, reactions: list, outcomes: list = None) -> dict:
    """Build a minimal FAERS case dict for testing."""
    return {
        "primaryid":        pid,
        "sex":              "M",
        "age":              45.0,
        "reporter_country": "US",
        "drugs":            drugs,
        "reactions":        reactions,
        "outcomes":         outcomes or ["OT"],
    }


def _build_synthetic_batch() -> tuple[list[dict], dict]:
    """
    Build a 77-record micro-batch with exactly 2 statistically significant pairs.

    Signal pairs (high PRR, low p-value):
      - ("WARFARIN",    "HAEMORRHAGE")    — 20 co-occurrences
      - ("METFORMIN",   "LACTIC ACIDOSIS") — 15 co-occurrences

    Noise pairs (appear < 3 times or have low PRR):
      - ("ASPIRIN",     "NAUSEA")          — only 2 cases (below MIN_CASES_THRESHOLD)
      - ("IBUPROFEN",   "HEADACHE")        — 10 cases but PRR = 1.2 (not significant)
      - ("ATORVASTATIN","MYALGIA")         — 10 cases but p-value = 0.12 (not significant)

    Total cases: 20 + 15 + 2 + 10 + 10 + 20 (padding) = 77
    """
    casos = []
    pid = 1000

    # Signal pair 1: WARFARIN + HAEMORRHAGE  (20 cases)
    for _ in range(20):
        casos.append(_make_caso(pid, ["WARFARIN"], ["HAEMORRHAGE"], ["LT"]))
        pid += 1

    # Signal pair 2: METFORMIN + LACTIC ACIDOSIS  (15 cases)
    for _ in range(15):
        casos.append(_make_caso(pid, ["METFORMIN"], ["LACTIC ACIDOSIS"], ["HO"]))
        pid += 1

    # Noise pair 1: ASPIRIN + NAUSEA  (only 2 cases — below MIN_CASES_THRESHOLD=3)
    for _ in range(2):
        casos.append(_make_caso(pid, ["ASPIRIN"], ["NAUSEA"], ["OT"]))
        pid += 1

    # Noise pair 2: IBUPROFEN + HEADACHE  (10 cases, PRR=1.2 — below PRR_THRESHOLD)
    for _ in range(10):
        casos.append(_make_caso(pid, ["IBUPROFEN"], ["HEADACHE"], ["OT"]))
        pid += 1

    # Noise pair 3: ATORVASTATIN + MYALGIA  (10 cases, p-value=0.12 — above PVALUE_THRESHOLD)
    for _ in range(10):
        casos.append(_make_caso(pid, ["ATORVASTATIN"], ["MYALGIA"], ["DS"]))
        pid += 1

    # Padding cases with no matching signal model entry  (20 cases)
    for _ in range(20):
        casos.append(_make_caso(pid, ["LISINOPRIL"], ["DIZZINESS"], ["OT"]))
        pid += 1

    assert len(casos) == 77, f"Expected 77 cases, got {len(casos)}"
    return casos


def _build_mock_signals() -> dict:
    """
    Build a mock modelo_senales dict that has:
      - WARFARIN + HAEMORRHAGE    → PRR=8.5,  chi2=22.0  → p-value≈0.000003  ✓ SIGNAL
      - METFORMIN + LACTIC ACIDOSIS → PRR=5.1, chi2=16.8 → p-value≈0.000041  ✓ SIGNAL
      - IBUPROFEN + HEADACHE      → PRR=1.2,  chi2=0.9   → p-value≈0.34      ✗ low PRR
      - ATORVASTATIN + MYALGIA    → PRR=3.0,  chi2=2.4   → p-value≈0.12      ✗ high p-value
      (ASPIRIN+NAUSEA and LISINOPRIL+DIZZINESS not in model at all)
    """
    return {
        ("WARFARIN",     "HAEMORRHAGE"):     {"prr": 8.5, "chi2": 22.0, "count_a": 120},
        ("METFORMIN",    "LACTIC ACIDOSIS"): {"prr": 5.1, "chi2": 16.8, "count_a":  85},
        ("IBUPROFEN",    "HEADACHE"):        {"prr": 1.2, "chi2":  0.9, "count_a": 300},
        ("ATORVASTATIN", "MYALGIA"):         {"prr": 3.0, "chi2":  2.4, "count_a":  60},
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def consumer():
    """
    Returns a FAERSConsumer instance with Kafka disabled and a mocked signal model.
    All file I/O (alerts file, logging) is also patched out.
    """
    with patch("faers_kafka.consumer.KafkaConsumer", side_effect=Exception("no broker")), \
         patch("faers_kafka.consumer.ALERTS_FILE", MagicMock()), \
         patch("builtins.open", MagicMock()), \
         patch("faers_kafka.consumer.logging"):

        from faers_kafka.consumer import FAERSConsumer

        c = FAERSConsumer(shared_queue=queue.Queue())
        # Replace the signal model with our controlled mock
        c.modelo_senales = _build_mock_signals()
        return c


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMicroBatchSignalDetection:
    """
    Tests for FAERSConsumer.procesar_micro_batch() with a 77-record batch
    containing exactly 2 true signal pairs.
    """

    def test_exactly_two_alerts_generated(self, consumer):
        """
        Core assertion: 77 input records → exactly 2 alerts.
        This catches the original inflation bug where 15,000+ alerts were emitted.
        """
        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)
        assert len(alerts) == 2, (
            f"Expected exactly 2 alerts, got {len(alerts)}. "
            f"Alert pairs: {[(a['drugname'], a['pt']) for a in alerts]}"
        )

    def test_alert_pairs_are_correct(self, consumer):
        """Alerts must correspond to the 2 true signal pairs, not noise pairs."""
        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)

        alert_pairs = {(a["drugname"], a["pt"]) for a in alerts}
        assert ("WARFARIN",  "HAEMORRHAGE")     in alert_pairs
        assert ("METFORMIN", "LACTIC ACIDOSIS") in alert_pairs

    def test_no_alert_for_below_threshold_pair(self, consumer):
        """ASPIRIN+NAUSEA has only 2 cases — must NOT generate an alert (N < 3)."""
        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)

        alert_pairs = {(a["drugname"], a["pt"]) for a in alerts}
        assert ("ASPIRIN", "NAUSEA") not in alert_pairs

    def test_no_alert_for_low_prr_pair(self, consumer):
        """IBUPROFEN+HEADACHE has PRR=1.2 — must NOT generate an alert (PRR ≤ 2.0)."""
        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)

        alert_pairs = {(a["drugname"], a["pt"]) for a in alerts}
        assert ("IBUPROFEN", "HEADACHE") not in alert_pairs

    def test_no_alert_for_high_pvalue_pair(self, consumer):
        """ATORVASTATIN+MYALGIA has p-value=0.12 — must NOT generate an alert (p ≥ 0.05)."""
        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)

        alert_pairs = {(a["drugname"], a["pt"]) for a in alerts}
        assert ("ATORVASTATIN", "MYALGIA") not in alert_pairs

    def test_no_alert_for_pair_not_in_model(self, consumer):
        """LISINOPRIL+DIZZINESS is not in the signal model — must NOT generate an alert."""
        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)

        alert_pairs = {(a["drugname"], a["pt"]) for a in alerts}
        assert ("LISINOPRIL", "DIZZINESS") not in alert_pairs

    def test_deduplication_within_batch(self, consumer):
        """
        Sending the same (drug, pt) pair 50 times in one batch must produce only
        1 alert for that pair — not 50.
        """
        from faers_kafka.consumer import MIN_CASES_THRESHOLD

        # Create a batch with a single repeated pair (50 rows)
        repeated_batch = [
            _make_caso(i, ["WARFARIN"], ["HAEMORRHAGE"], ["LT"])
            for i in range(50)
        ]
        alerts = consumer.procesar_micro_batch(repeated_batch)
        warfarin_alerts = [a for a in alerts if a["drugname"] == "WARFARIN" and a["pt"] == "HAEMORRHAGE"]
        assert len(warfarin_alerts) == 1, (
            f"Expected 1 alert for WARFARIN+HAEMORRHAGE, got {len(warfarin_alerts)}"
        )

    def test_alert_payload_has_pvalue_field(self, consumer):
        """Alert payload must include p_value (real p-value, not chi2 statistic)."""
        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)

        for alert in alerts:
            assert "p_value" in alert, "Alert is missing the p_value field."
            assert alert["p_value"] is not None
            assert 0.0 < alert["p_value"] < 0.05, (
                f"p_value={alert['p_value']} should be < 0.05 for a confirmed signal."
            )

    def test_alert_payload_has_prr_above_threshold(self, consumer):
        """Alert payload must include PRR and it must be > PRR_THRESHOLD (2.0)."""
        from faers_kafka.consumer import PRR_THRESHOLD

        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)

        for alert in alerts:
            assert "prr" in alert
            assert alert["prr"] > PRR_THRESHOLD, (
                f"PRR={alert['prr']} should be > {PRR_THRESHOLD}"
            )

    def test_alert_payload_has_n_cases_in_batch(self, consumer):
        """Alert payload must report the number of batch cases for the pair."""
        batch = _build_synthetic_batch()
        alerts = consumer.procesar_micro_batch(batch)

        for alert in alerts:
            assert "n_cases_in_batch" in alert
            assert alert["n_cases_in_batch"] >= 3, (
                f"n_cases_in_batch={alert['n_cases_in_batch']} must be >= MIN_CASES_THRESHOLD (3)"
            )

    def test_stats_delta_is_correct_after_batch(self, consumer):
        """
        After processing the batch, self.stats['total_alertas'] must equal 2
        (assuming a fresh consumer starting at 0).
        """
        batch = _build_synthetic_batch()
        consumer.procesar_micro_batch(batch)

        assert consumer.stats["total_alertas"] == 2, (
            f"Expected stats['total_alertas']=2, got {consumer.stats['total_alertas']}"
        )
        assert consumer.stats["total_consumidos"] == 77, (
            f"Expected stats['total_consumidos']=77, got {consumer.stats['total_consumidos']}"
        )

    def test_stats_reset_between_batches(self, consumer):
        """
        Processing a second batch with 0 signals must not add to the alert count
        from the first batch — counters are deltas, not re-counted globally wrong.
        """
        batch = _build_synthetic_batch()
        consumer.procesar_micro_batch(batch)
        alerts_after_first = consumer.stats["total_alertas"]

        # Second batch: all noise (no signals)
        noise_batch = [
            _make_caso(i + 9000, ["ASPIRIN"], ["HEADACHE"], ["OT"])
            for i in range(5)
        ]
        consumer.procesar_micro_batch(noise_batch)

        # total_alertas must remain at alerts_after_first (no new alerts added)
        assert consumer.stats["total_alertas"] == alerts_after_first, (
            "Noise batch should not change total_alertas."
        )

    def test_empty_batch_produces_no_alerts(self, consumer):
        """An empty batch must return an empty alert list."""
        alerts = consumer.procesar_micro_batch([])
        assert alerts == []

    def test_single_case_below_threshold(self, consumer):
        """A batch with a single case cannot meet MIN_CASES_THRESHOLD — no alerts."""
        single_batch = [_make_caso(1, ["WARFARIN"], ["HAEMORRHAGE"], ["LT"])]
        alerts = consumer.procesar_micro_batch(single_batch)
        assert alerts == [], (
            "A single-case batch should produce no alerts (below MIN_CASES_THRESHOLD=3)."
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running directly: python tests/test_consumer_signal_detection.py
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
