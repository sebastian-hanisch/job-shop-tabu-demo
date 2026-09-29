"""Konstanten der Job-Shop-Tabu-Search-Demo: beide Vehikel (Neutral, Werkstatt/Logistik), Regler, Messreihen-Seeds.
Dasselbe Job-Shop-Modell wie `job-shop-demo` (Stück 8) und `shifting-bottleneck-demo` (Stück 9)."""

N_MIN, N_MAX, DEFAULT_N, N_STEP = 2, 20, 10, 1
M_MIN, M_MAX, DEFAULT_M = 2, 6, 4
SEED_MAX = 999999
DEFAULT_SEED = 60
SWEEP_SEEDS = tuple(range(100000, 100005))

# Bearbeitungszeiten
P_MIN, P_MAX = 1, 100

# CP-SAT exakte Gegenprobe (dasselbe Kreis-Modell wie Stück 8/9).
EXACT_MAX_N = 8
EXACT_TIME_LIMIT_SECONDS = 15.0

# Tabu Search selbst nutzt KEIN CP-SAT (reine Graph-/Kombinatorik-Operationen: kritischer Pfad, Blöcke, N5-Züge) -
# deshalb kein Worker-/Nichtdeterminismus-Thema wie in Stück 9.
DEFAULT_MAX_ITERS, MAX_ITERS_MIN, MAX_ITERS_MAX = 200, 20, 1000
DEFAULT_TENURE, TENURE_MIN, TENURE_MAX = 8, 1, 40

# --- Vehikel B "Werkstatt/Logistik" ---------------------------------------------------------------------------
N_FAMILIES_MIN, N_FAMILIES_MAX, DEFAULT_N_FAMILIES = 2, 6, 3
SETUP_TIME_MIN, SETUP_TIME_MAX, DEFAULT_SETUP_TIME = 0, 60, 15

VEHICLE_LABELS = {"neutral": "Neutral", "logistik": "Werkstatt/Logistik"}
DEFAULT_VEHICLE = "neutral"


def _preset(n=DEFAULT_N, m=DEFAULT_M, vehicle=DEFAULT_VEHICLE, setup_time=DEFAULT_SETUP_TIME, n_families=DEFAULT_N_FAMILIES,
            max_iters=DEFAULT_MAX_ITERS, tenure=DEFAULT_TENURE):
    return {"n": n, "m": m, "seed": DEFAULT_SEED, "vehicle": vehicle, "setup_time": setup_time, "n_families": n_families,
            "max_iters": max_iters, "tenure": tenure}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Kleine Instanz (CP-SAT sichtbar)": _preset(n=EXACT_MAX_N),
    "Große Instanz (Skalierung)": _preset(n=N_MAX),
    "Werkstatt/Logistik-Vehikel": _preset(vehicle="logistik"),
    "Hohe Rüstlast (Werkstatt)": _preset(vehicle="logistik", setup_time=SETUP_TIME_MAX),
}
# Werte in PRESET_HELP nach der Messreihe (jst_evaluation.run_config) final eingetragen.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": f"{DEFAULT_N} Aufträge auf {DEFAULT_M} Maschinen: Tabu Search verbessert die MWKR-Startlösung (Stück 8) über den kritischen Pfad.",
    "Kleine Instanz (CP-SAT sichtbar)": f"{EXACT_MAX_N} Aufträge: hier löst CP-SAT (OR-Tools) das Problem exakt mit.",
    "Große Instanz (Skalierung)": f"{N_MAX} Aufträge: Tabu Search bleibt praktikabel, eine exakte Lösung wäre bei dieser Größe aussichtslos.",
    "Werkstatt/Logistik-Vehikel": "Dieselben Aufträge, aber in Familien mit Rüstzeit beim Wechsel je Maschine - der kritische Pfad berücksichtigt die Rüstzeit mit.",
    "Hohe Rüstlast (Werkstatt)": f"Rüstzeit {SETUP_TIME_MAX} Minuten je Familienwechsel: eine härtere Ausgangslage für die Verbesserung.",
}
