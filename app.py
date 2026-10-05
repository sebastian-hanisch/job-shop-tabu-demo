"""Job-Shop-Tabu-Search - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Konzepte-Linie "Klassische Scheduling-Theorie", das zweite SOTA-Stück: statt einmal gut zu
KONSTRUIEREN (Shifting Bottleneck, Stück 9) verbessert das Tabu-Search-Verfahren von Nowicki & Smutnicki (1996)
eine gegebene Startlösung (MWKR, Stück 8) iterativ über eine gezielte, kleine Nachbarschaft (N5): pro Block des
kritischen Pfads wird nur das erste oder letzte Operationspaar vertauscht. Eine Tabu-Liste verhindert Zyklen.
Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import jst_constants as C
from jst_evaluation import Settings, SWEEP_LABELS, analyse, setup_gap_sweep, start_comparison, sweep, timing_sweep
from jst_presets import KEPT, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, seed_widget, sync_query_params
from jst_visualization import build_jobs_chart, build_machine_finish_comparison, build_machine_gantt, build_setup_gap, build_start_comparison, build_sweep, build_timing

st.set_page_config(page_title="Job-Shop-Tabu-Search – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _start_comparison():
    return start_comparison()


@st.cache_data(show_spinner=False)
def _timing(m):
    return timing_sweep(m=m)


@st.cache_data(show_spinner=False)
def _setup_gap_sweep(n, m, n_families):
    return setup_gap_sweep(n=n, m=m, n_families=n_families)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


def _fmt_pct(x):
    """Vorzeichen-korrekt: `+18.7 %` (MWKR-Start schlechter als Tabu Search) oder `-x %` (kann bei einer
    besonders schwierigen Instanz theoretisch passieren - kein Fehler, siehe 🚧-Abschnitt)."""
    return f"{x:+.1f} %"


st.title("🔄 Job-Shop-Tabu-Search – kritischer Pfad, gezielt verbessert")
st.markdown(
    r"""
**Zweites SOTA-Stück dieser Linie.** Shifting Bottleneck (Stück 9) konstruiert EINMAL gut. **Tabu Search**
(Nowicki & Smutnicki 1996) geht anders vor: es startet bei einer gegebenen Lösung (hier MWKR, Stück 8) und
VERBESSERT sie iterativ - über eine gezielte, winzige Nachbarschaft statt aller denkbaren Vertauschungen. Auf dem
**kritischen Pfad** (dem Weg ohne jeden Puffer) wird in **Blöcke** zerlegt (maximale Teilfolgen auf derselben
Maschine); pro Block wird nur das ERSTE oder LETZTE Operationspaar vertauscht (**N5-Nachbarschaft**) - eine
Tabu-Liste verhindert das sofortige Zurücktauschen.
"""
)
st.caption(
    "Zehntes Stück der Konzepte-Linie „Klassische Scheduling-Theorie“ - dasselbe Job-Shop-Modell wie Stück 8/9. "
    "Zwei Vehikel: **Neutral** und **Werkstatt/Logistik** (Familien mit Rüstzeit beim Wechsel je Maschine) - "
    "der Umschalter ist in der Seitenleiste."
)

with st.expander("So funktioniert Tabu Search mit der N5-Nachbarschaft", expanded=True):
    st.markdown(
        r"""
1. **Kritischen Pfad finden.** Der Weg vom ersten zum letzten Operationsende ohne jeden Puffer - jede Kante ist
   "straff" (Nachfolger beginnt GENAU dann, wenn Vorgänger endet, plus Rüstzeit auf dem Werkstatt-Vehikel).
2. **In Blöcke zerlegen.** Maximale Teilfolgen des kritischen Pfads, die auf DERSELBEN Maschine liegen.
3. **N5-Züge bilden.** Pro Block mit ≥2 Operationen das ERSTE oder LETZTE Paar vertauschen - AUSSER dem ersten
   Paar des allerersten Blocks und dem letzten Paar des allerletzten Blocks (diese können Cmax ohne Rüstzeiten
   beweisbar nicht verbessern, Nowicki & Smutnicki 1996; mit Rüstzeit auf dem Werkstatt-Vehikel entfällt die Ausnahme).
4. **Besten erlaubten Zug wählen.** Tabu-Züge werden übersprungen, AUSSER eine Aspiration (neues Gesamtoptimum)
   erlaubt es trotzdem - sind wirklich alle Züge tabu, wird der am wenigsten schlechte trotzdem gemacht.
5. **Wiederholen**, bis das Budget erschöpft ist. Die BESTE je gefundene Lösung zählt, nicht die letzte.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Aufträge", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
                        help=f"Anzahl der Aufträge. Bis {C.EXACT_MAX_N} löst CP-SAT das volle Problem exakt mit.")
    m_machines = st.slider("Maschinen", *bounds("m_slider"), key="m_slider",
                            help="Jeder Auftrag besucht jede Maschine genau einmal, aber in eigener Reihenfolge.")
    vehicle = st.radio("Vehikel", list(C.VEHICLE_LABELS), key="vehicle_radio", format_func=lambda k: C.VEHICLE_LABELS[k],
                        help="Neutral: nur Bearbeitungszeiten. Werkstatt/Logistik: dieselben Aufträge, zusätzlich in Familien mit Rüstzeit beim Wechsel je Maschine.")
    if vehicle == "logistik":
        seed_widget("setup_time_slider")
        setup_time = st.slider("Rüstzeit je Familienwechsel (Minuten)", *bounds("setup_time_slider"), key="setup_time_slider",
                                help="0 Minuten kollabiert strukturell exakt zum neutralen Vehikel (siehe Test/Messreihe).")
        st.session_state[KEPT["setup_time_slider"]] = setup_time
        seed_widget("n_families_slider")
        n_families = st.slider("Auftragsfamilien", *bounds("n_families_slider"), key="n_families_slider",
                                help="Mehr Familien bei gleicher Auftragszahl bedeuten mehr Wechsel und damit mehr Rüstzeit insgesamt.")
        st.session_state[KEPT["n_families_slider"]] = n_families
    else:
        setup_time = int(st.session_state.get(KEPT["setup_time_slider"], C.DEFAULT_SETUP_TIME))
        n_families = int(st.session_state.get(KEPT["n_families_slider"], C.DEFAULT_N_FAMILIES))
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für Routing und Bearbeitungszeiten.")
    max_iters = st.slider("Budget (Iterationen)", *bounds("max_iters_slider"), key="max_iters_slider",
                           help="Maximale Anzahl an Tabu-Search-Runden.")
    tenure = st.slider("Tabu-Tenure", *bounds("tenure_slider"), key="tenure_slider",
                        help="Wie viele Runden ein gerade gemachter Zug gesperrt bleibt (außer bei Aspiration).")

sync_query_params({"n_slider": int(n_jobs), "m_slider": int(m_machines), "seed_input": int(seed),
                    "vehicle_radio": vehicle, "setup_time_slider": int(setup_time), "n_families_slider": int(n_families),
                    "max_iters_slider": int(max_iters), "tenure_slider": int(tenure)})

settings = Settings(int(n_jobs), int(m_machines), int(seed), vehicle=vehicle, setup_time=int(setup_time),
                     n_families=int(n_families), max_iters=int(max_iters), tenure=int(tenure))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
routing, proc = inst.routing, inst.proc
data_key = settings

# --- Tabu Search in Aktion -----------------------------------------------------------------------------------

st.markdown("## 🎯 Tabu Search in Aktion")
STEP_LABELS = {1: "1 · Aufträge", 2: "2 · Kritischer Pfad & Blöcke", 3: "3 · Ergebnis"}
if "jst_step" not in st.session_state or st.session_state.get("jst_step_owner") != data_key:
    st.session_state["jst_step"] = 1
    st.session_state["jst_step_owner"] = data_key
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="jst_step", format_func=lambda s: STEP_LABELS[s])

trace = a.tabu.trace or [{"it": 0, "cmax": a.mwkr.cmax, "start": a.mwkr.start, "end": a.mwkr.end, "critical_ops": set()}]
if step == 2:
    imp_col, _ = st.columns([5, 2])
    with imp_col:
        upto = st.slider("Verbesserung", 0, len(trace) - 1, value=len(trace) - 1, key="jst_upto",
                          help="0 = Startlösung (MWKR), jeder weitere Schritt eine gefundene Verbesserung.")
else:
    upto = len(trace) - 1

view_slot = st.empty()
with view_slot.container():
    if step == 1:
        st.markdown(f"**{n_jobs} Aufträge, unsortiert** (gestapelt in eigener Maschinenreihenfolge, Farbe nach Maschine)")
        st.plotly_chart(build_jobs_chart(routing, proc), width="stretch", key="s1_jobs")
    elif step == 2:
        snap = trace[upto]
        st.markdown(f"**Cmax {_fmt_int(snap['cmax'])} nach {upto} von {len(trace) - 1} gefundenen Verbesserungen** (Farbe nach Auftrag, kritischer Pfad schwarz umrandet)")
        st.plotly_chart(build_machine_gantt(routing, proc, snap["start"], snap["end"], int(m_machines), critical_ops=snap["critical_ops"]), width="stretch", key=f"s2_sched_{upto}")
    else:
        st.markdown("**Fertigstellung je Maschine: Tabu Search gegen den MWKR-Start**")
        st.plotly_chart(build_machine_finish_comparison(a.tabu.machine_finish, a.mwkr.machine_finish), width="stretch", key="s3_finish")

if step == 1:
    st.caption(f"Bearbeitungszeiten zwischen {int(proc.min())} und {int(proc.max())} Minuten (Seed {seed}). Jeder Auftrag besucht jede Maschine genau einmal, aber in eigener Reihenfolge.")
elif step == 2:
    st.caption("Schwarz umrandete Balken liegen auf dem kritischen Pfad dieser Zwischenlösung - genau diese Operationen liefern die N5-Züge für die nächste Verbesserung.")
else:
    st.caption(f"Tabu Search: Cmax {_fmt_int(a.tabu.cmax)}. MWKR-Start: {_fmt_int(a.mwkr.cmax)} (Differenz {_fmt_pct(a.gap_mwkr)}). Die gestrichelten Linien markieren jeweils die höchste Last (= Cmax).")

st.markdown("---")

# --- Ergebnis ----------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Verbesserung bringt")
vehicle_note = " Auf dem Werkstatt/Logistik-Vehikel zählt die Rüstzeit je Maschine mit - der kritische Pfad berücksichtigt sie." if vehicle == "logistik" else ""
st.caption(f"**Abstand:** Cmax des MWKR-Starts gegenüber Tabu Search in Prozent.{vehicle_note}")
m1, m2, m3 = st.columns(3)
m1.metric("Tabu Search (Cmax)", _fmt_int(a.tabu.cmax), help="Die Zielgröße: Gesamtdurchlaufzeit nach der Tabu-Suche, auf dem gewählten Vehikel.")
m2.metric("MWKR-Start (Stück 8)", _fmt_pct(a.gap_mwkr), delta_color="off", help="Die Startlösung selbst, unverändert - der Abstand zeigt, was die Suche daraus gemacht hat.")
if a.optimal is not None and a.optimal_proven:
    m3.metric("CP-SAT (exakte Gegenprobe)", "trifft das Optimum exakt" if a.tabu_matches_optimum else f"{a.tabu_ratio_to_optimum:.3f}× Optimum", delta_color="off",
              help="OR-Tools CP-SAT hat diese Instanz bewiesen exakt gelöst (volles Modell, nicht nur ein Teilproblem).")
elif a.optimal is not None:
    m3.metric("CP-SAT", "Zeitlimit erreicht", delta_color="off", help="CP-SAT hat innerhalb des Zeitlimits keine bewiesen optimale Lösung gefunden.")
else:
    m3.metric("CP-SAT", f"erst ab n ≤ {C.EXACT_MAX_N}", delta_color="off", help="Bei dieser Größe wäre eine exakte Lösung des vollen Modells aussichtslos.")

if a.gap_mwkr < 0:
    st.warning(f"⚠️ Unerwartet: Tabu Search liegt hier über dem MWKR-Start (dessen Cmax liegt {abs(a.gap_mwkr):.1f} % unter dem von Tabu Search). Das darf nicht vorkommen - Tabu Search gibt immer die beste je gefundene Lösung zurück, also höchstens den Start selbst. Bitte als Fehler melden.")
else:
    st.success(f"✅ Tabu Search ist besser als der MWKR-Start: dessen Cmax liegt {a.gap_mwkr:.1f} % darüber (Tabu Search verkürzt Cmax um {100 * a.gap_mwkr / (100 + a.gap_mwkr):.1f} %).")

st.markdown("---")

# --- Sweeps --------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Vorsprung von der Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Instanzen berechnen (dauert wenige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    rows_sweep = _sweep(sweep_param, Settings())
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004). Die gepunktete Nulllinie markiert Gleichstand mit dem MWKR-Start.")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------

st.subheader("🔬 Zählt ein besserer Start? MWKR (Stück 8) gegen Shifting Bottleneck (Stück 9)")
if st.button("Beide Startlösungen über n = 8 bis 20 vergleichen (dauert einige Sekunden)", key="start_cmp_start"):
    st.session_state["start_cmp_on"] = True
if st.session_state.get("start_cmp_on"):
    rows_sc = _start_comparison()
    st.plotly_chart(build_start_comparison(rows_sc), width="stretch", key="start_cmp_chart")
    last = rows_sc[-1]
    st.caption(f"Überraschender Befund: bei den meisten Instanzen liegen beide Linien fast übereinander (Gleichstand-Quote bis zu {max(r['tie_rate'] for r in rows_sc)*100:.0f} %) - anders als bei der Schwester-Tabu-Demo für TSP im Trajektorien-Metaheuristiken-Portfolio, wo ein guter Start stark zählt. Die N5-Nachbarschaft scheint robust genug, um einen schwächeren Start selbst auszugleichen.")

st.markdown("---")

st.subheader("🔬 Wie teuer ist eine exakte Lösung wirklich?")
if st.button("Rechenzeit für n = 2 bis 8 messen (dauert etwa 1 Sekunde)", key="timing_start"):
    st.session_state["timing_on"] = True
if st.session_state.get("timing_on"):
    rows_t = _timing(int(m_machines))
    st.plotly_chart(build_timing(rows_t), width="stretch", key="timing_chart")
    last = rows_t[-1]
    st.caption(f"Bei {last['value']} Aufträgen braucht das volle CP-SAT-Modell bereits {last['exact_seconds']*1000:.0f} ms, Tabu Search (reine Graph-/Kombinatorik-Operationen) {last['tabu_seconds']*1000:.1f} ms.")

st.markdown("---")

st.subheader("🔬 Werkstatt/Logistik: bleibt der Vorsprung, wenn Rüstzeiten dazukommen?")
if st.button("Rüstzeit von 0 bis 60 Minuten durchfahren (dauert wenige Sekunden)", key="setup_start"):
    st.session_state["setup_on"] = True
if st.session_state.get("setup_on"):
    rows_s = _setup_gap_sweep(min(int(n_jobs), C.EXACT_MAX_N), int(m_machines), int(n_families))
    st.plotly_chart(build_setup_gap(rows_s), width="stretch", key="setup_chart")
    st.caption("Tabu Search (Start MWKR, rüstzeitbewusster kritischer Pfad) verglichen mit der echten Optimallösung MIT Rüstzeiten (CP-SAT, deshalb kleine Instanz).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die N5-Nachbarschaft findet immer einen Zug** | Liegt der GESAMTE kritische Pfad auf EINER Maschine (ein einziger Block), bietet N5 nach der literaturüblichen Ausnahmeregel (Vehikel Neutral, Werkstatt ohne Rüstzeit) KEINEN Zug an - die Suche steht sofort still. | Kein direkter Nachfolger in dieser Linie |
| **Ein besserer Start hilft immer** | Gemessen NICHT bestätigt (siehe 🔬 oben) - MWKR- und Shifting-Bottleneck-Start konvergieren bei diesem Budget fast immer zum selben Ergebnis. | Kein direkter Nachfolger in dieser Linie |
| **Eine feste Startlösung reicht** | Für sehr harte Instanzen kombinieren professionelle Löser mehrere Startlösungen oder Neustarts. | **RCPSP** (letztes Stück, wachsender Endpunkt) |
"""
)
st.caption(
    "Zehntes Stück der Linie „Klassische Scheduling-Theorie“, zweites SOTA-Stück. Verwandt: die Trajektorien-"
    "Metaheuristiken-Linie (`tabu-search-demo`, Tabu Search für TSP mit voller Nachbarschaft statt N5)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem** (Job Shop, $C_{\max}$): dasselbe Modell wie Stück 8/9 - $n$ Aufträge, $m$ Maschinen, jeder Auftrag mit
eigener Reihenfolge über die Maschinen.

**Kritischer Pfad.** Die Folge von Operationen vom Start bis zur Fertigstellung mit $C_{\max}$, bei der JEDE Kante
straff ist: $\text{start}(\text{Nachfolger}) = \text{end}(\text{Vorgänger})$ (plus Rüstzeit bei Vehikel B).

**Blöcke.** Maximale Teilfolgen des kritischen Pfads auf DERSELBEN Maschine.

**N5-Nachbarschaft** (Nowicki & Smutnicki 1996). Für jeden Block $B$ mit $|B| \geq 2$: vertausche die ersten
beiden ODER die letzten beiden Operationen - außer beim ersten Paar des ERSTEN Blocks und beim letzten Paar des
LETZTEN Blocks (diese Züge können $C_{\max}$ ohne Rüstzeiten beweisbar nicht verkleinern; bei Vehikel B mit Rüstzeit werden sie mit angeboten).

**Tabu-Liste.** Ein Zug $(k, \text{neue Reihenfolge})$ ist für `tenure` Runden gesperrt, nachdem er gemacht wurde
- außer eine Aspiration (neues Gesamtoptimum) erlaubt ihn trotzdem.

**CP-SAT-Modell** (`solve_exact`): dasselbe Kreis-Modell wie Stück 8/9, hier als exakte Gegenprobe auf das VOLLE
Problem.

Implementiert in `jst_algorithm.py` (kritischer Pfad, Blöcke, N5-Nachbarschaft, Tabu-Search-Hauptschleife,
Giffler-Thompson/MWKR als Startlösung, CP-SAT), `jst_shifting_bottleneck.py` (zweite Startlösung für das
Start-Vergleichs-Experiment), `jst_scenario.py`/`jst_scenario_logistik.py` (die zwei Vehikel), `jst_evaluation.py`
(Kennzahlen, Sweep, Start-Vergleich, Timing-Messreihe, Rüstzeit-Härtetest).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Scheduling-Theorie: SPT bis RCPSP](https://sebastianhanisch.net/konzepte-klassische-scheduling-theorie.html)."
)
