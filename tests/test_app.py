"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Würfel-Knopf, Permalink-Grenzen,
Vehikel-Umschalter, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import jst_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(jst_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if jst_step != 1:
        at.select_slider(key="jst_step").set_value(jst_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "Tabu Search (Cmax)") == "595"
    assert _metric(at, "MWKR-Start (Stück 8)") == "+18.7 %"
    assert any("Tabu Search ist" in s.value for s in at.success)


def test_switching_to_the_logistik_vehicle_actually_changes_the_main_metric():
    """Regressionsschutz für dieselbe Lücke, die in Stück 1-3 dieser Linie gefunden wurde: der Vehikel-
    Umschalter muss die HAUPT-Kennzahl ändern, nicht nur eine separate Box."""
    at_neutral = _run(n_slider=10, seed_input=7, vehicle_radio="neutral")
    at_logistik = _run(n_slider=10, seed_input=7, vehicle_radio="logistik", setup_time_slider=60, n_families_slider=2)
    _ok(at_neutral)
    _ok(at_logistik)
    assert _metric(at_neutral, "Tabu Search (Cmax)") != _metric(at_logistik, "Tabu Search (Cmax)")


def test_gap_mwkr_is_never_negative_by_construction():
    """ANDERS als Stück 7-9: Tabu Search gibt immer die BESTE je gefundene Lösung zurück, nie schlechter als
    die eigene Startlösung - eine bewiesene, nicht nur gemessene Garantie."""
    for seed in (2, 7, 42):
        at = _run(n_slider=6, m_slider=3, seed_input=seed, vehicle_radio="neutral")
        _ok(at)
        assert not any(w for w in at.warning if "Unerwartet: Tabu Search liegt hier über dem MWKR-Start" in w.value)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["n_slider"] == p["n"] and at.session_state["vehicle_radio"] == p["vehicle"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("vehicle", ["neutral", "logistik"])
def test_every_step_runs_on_both_vehicles(step, vehicle):
    at = _run(n_slider=8, vehicle_radio=vehicle, jst_step=step)
    _ok(at)
    assert at.session_state["jst_step"] == step


def test_step_two_has_the_upto_slider_defaulting_to_the_last_improvement():
    at = _run(n_slider=8, m_slider=3, jst_step=2)
    _ok(at)
    sl = next(s for s in at.slider if s.key == "jst_upto")
    assert sl.value == sl.max


def test_exact_limit_is_respected_in_the_metric():
    at = _run(n_slider=C.EXACT_MAX_N)
    _ok(at)
    proven_metric = next((m for m in at.metric if m.label == "CP-SAT (exakte Gegenprobe)"), None)
    timeout_metric = next((m for m in at.metric if m.label == "CP-SAT" and m.value == "Zeitlimit erreicht"), None)
    assert proven_metric is not None or timeout_metric is not None
    at2 = _run(n_slider=C.EXACT_MAX_N + 1)
    _ok(at2)
    assert "erst ab n" in _metric(at2, "CP-SAT")


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN), dict(n_slider=C.N_MAX), dict(m_slider=C.M_MIN), dict(m_slider=C.M_MAX),
                                 dict(vehicle_radio="logistik", setup_time_slider=C.SETUP_TIME_MIN),
                                 dict(vehicle_radio="logistik", setup_time_slider=C.SETUP_TIME_MAX),
                                 dict(vehicle_radio="logistik", n_families_slider=C.N_FAMILIES_MIN),
                                 dict(max_iters_slider=C.MAX_ITERS_MIN), dict(max_iters_slider=C.MAX_ITERS_MAX),
                                 dict(tenure_slider=C.TENURE_MIN), dict(tenure_slider=C.TENURE_MAX)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["m"] = "9999"
    at.query_params["vehicle"] = "logistik"
    at.query_params["setup"] = "9999"
    at.query_params["iters"] = "9999"
    at.query_params["tenure"] = "9999"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["m_slider"] == C.M_MAX
    assert at.session_state["vehicle_radio"] == "logistik" and at.session_state["setup_time_slider"] == C.SETUP_TIME_MAX
    assert at.session_state["max_iters_slider"] == C.MAX_ITERS_MAX and at.session_state["tenure_slider"] == C.TENURE_MAX


def test_permalink_ignores_an_invalid_vehicle():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["vehicle"] = "nicht_vorhanden"
    at.run()
    _ok(at)
    assert at.session_state["vehicle_radio"] == C.DEFAULT_VEHICLE


@pytest.mark.parametrize("param", ["n", "m"])
def test_sweeps_run_on_demand(param):
    at = _run(n_slider=8)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand():
    at = _run(n_slider=6, vehicle_radio="logistik")
    for key, flag in (("start_cmp_start", "start_cmp_on"), ("timing_start", "timing_on"), ("setup_start", "setup_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Die N5-Nachbarschaft findet immer einen Zug" in m.value for m in at.markdown)
