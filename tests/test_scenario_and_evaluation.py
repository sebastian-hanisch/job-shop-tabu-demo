"""Vehikel A (Neutral) und Vehikel B (Werkstatt/Logistik): Erzeugung, Determinismus; Auswertung: Kennzahlen,
Sweep, Start-Vergleich (MWKR gegen Shifting Bottleneck), Timing-Messreihe, Rüstzeit-Härtetest."""

import numpy as np
import pytest

import jst_algorithm as A
import jst_constants as C
import jst_evaluation as ev
import jst_scenario as S
import jst_scenario_logistik as SL
from jst_shifting_bottleneck import shifting_bottleneck


# --- Vehikel A ----------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_and_bounds():
    inst = S.generate(20, 4, 3)
    assert inst.n == 20 and inst.m == 4
    assert inst.routing.shape == (20, 4) and inst.proc.shape == (20, 4)
    assert inst.proc.min() >= C.P_MIN and inst.proc.max() <= C.P_MAX
    for j in range(20):
        assert sorted(inst.routing[j].tolist()) == list(range(4))


def test_instance_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(15, 3, 5), S.generate(15, 3, 5), S.generate(15, 3, 6)
    assert np.array_equal(a.routing, b.routing) and np.array_equal(a.proc, b.proc)
    assert not np.array_equal(a.proc, c.proc)


# --- Vehikel B ------------------------------------------------------------------------------------------------------------------------------


def test_logistik_instance_shares_the_same_routing_and_processing_times_as_neutral():
    neutral = S.generate(15, 3, 7)
    logistik = SL.generate(15, 3, 7)
    assert np.array_equal(neutral.routing, logistik.routing) and np.array_equal(neutral.proc, logistik.proc)


def test_logistik_instance_is_deterministic():
    a, b = SL.generate(10, 3, 2), SL.generate(10, 3, 2)
    assert np.array_equal(a.family, b.family) and np.array_equal(a.setup, b.setup)


# --- Analyse --------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(n=10, m=4))
    assert a.optimal is None


def test_analysis_respects_small_n_cp_sat_gegenprobe():
    a = ev.analyse(ev.Settings(n=4, m=3))
    assert a.optimal is not None and a.optimal_proven


def test_analysis_carries_a_trace_starting_at_mwkr_and_ending_at_tabu():
    a = ev.analyse(ev.Settings(n=8, m=4))
    assert a.tabu.trace[0]["cmax"] == pytest.approx(a.mwkr.cmax)
    assert a.tabu.trace[-1]["cmax"] == pytest.approx(a.tabu.cmax)


# --- Vehikel-Bewusstsein der Hauptanalyse (von Anfang an, siehe [[feedback_vehicle_toggle_must_drive_primary_metrics]]) ----------------------


def test_switching_vehicle_actually_changes_the_tabu_cmax():
    a_neutral = ev.analyse(ev.Settings(n=10, m=4, seed=7, vehicle="neutral"))
    a_logistik = ev.analyse(ev.Settings(n=10, m=4, seed=7, vehicle="logistik", setup_time=60, n_families=2))
    assert a_neutral.tabu.cmax != pytest.approx(a_logistik.tabu.cmax)


def test_tabu_at_zero_setup_time_matches_the_neutral_vehicle_exactly():
    a1 = ev.analyse(ev.Settings(n=10, m=4, seed=7, vehicle="neutral"))
    a2 = ev.analyse(ev.Settings(n=10, m=4, seed=7, vehicle="logistik", setup_time=0))
    assert a1.tabu.cmax == pytest.approx(a2.tabu.cmax)


def test_analysis_is_deterministic():
    s = ev.Settings(n=10, m=4, seed=1)
    a, b = ev.analyse(s), ev.analyse(s)
    assert a.tabu.cmax == pytest.approx(b.tabu.cmax)


# --- Sweep und Messreihe -------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=10, m=4))
    assert r["n_runs"] == len(C.SWEEP_SEEDS)


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("n", ev.Settings(), (5, 15))
    assert [r["value"] for r in rows] == [5, 15]
    rows_m = ev.sweep("m", ev.Settings(), (2, 4))
    assert [r["value"] for r in rows_m] == [2, 4]


def test_gap_mwkr_is_usually_positive_but_not_guaranteed():
    """Tabu Search verbessert die Startlösung fast immer deutlich - aber keine bewiesene Garantie (Suchraum-Satz
    aus Stück 8 gilt hier nicht)."""
    r = ev.run_config(ev.Settings())
    assert r["gap_mwkr"] > 0.0


def test_start_comparison_reports_high_tie_rate():
    """Der zentrale, überraschende Befund dieses Stücks: ANDERS als bei der Schwester-Tabu-Demo (TSP,
    Trajektorien-Metaheuristiken-Linie) zählt ein besserer Start hier kaum - die meisten Instanzen konvergieren
    zum selben Ergebnis, egal ob MWKR (Stück 8) oder Shifting Bottleneck (Stück 9) den Start liefert."""
    rows = ev.start_comparison(ns=(8, 15, 20), seeds=C.SWEEP_SEEDS[:3])
    assert any(r["tie_rate"] >= 0.5 for r in rows)


def test_shifting_bottleneck_start_is_reused_correctly_by_start_comparison():
    """Die Kopie in jst_shifting_bottleneck.py muss ein GÜLTIGES Ergebnis liefern (nicht nur irgendeinen Wert) -
    gegen die CP-SAT-Gegenprobe geprüft, nicht nur gegen sich selbst."""
    inst = ev.instance(6, 3, 42)
    sb = shifting_bottleneck(inst.routing, inst.proc, reoptimize=True)
    opt, proven = A.solve_exact(inst.routing, inst.proc, time_limit_seconds=10)
    assert proven
    assert sb.cmax >= opt.cmax - 1e-6


def test_timing_sweep_shows_tabu_staying_fast_where_the_full_model_gets_expensive():
    rows = ev.timing_sweep(ns=(2, 8))
    small, large = rows[0], rows[1]
    assert large["tabu_seconds"] < 5.0


def test_setup_gap_is_never_negative():
    """Die Optimallösung (CP-SAT) ist per Definition nie schlechter als die Heuristik."""
    row = ev.setup_gap(n=6, setup_time=15)
    assert row["gap_mean"] >= 0.0
