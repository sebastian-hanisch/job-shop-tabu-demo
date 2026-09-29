"""Auswertung der Job-Shop-Tabu-Search-Demo: Tabu Search (Start MWKR, Stück 8) gegen die eigene Startlösung,
gegen CP-SAT als exakte Gegenprobe (nur kleine n), das Vehikel-B-Experiment (Rüstzeit-Härtetest) und das
zentrale Experiment dieses Stücks - zählt ein besserer Start (Shifting Bottleneck, Stück 9) bei fester
Budget-Grenze? Überraschender, gemessener Befund (siehe README): kaum - anders als bei der Schwester-Tabu-Demo
im Trajektorien-Metaheuristiken-Portfolio (dort zählt der Start stark)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import jst_algorithm as A
import jst_constants as C
import jst_scenario as S
import jst_scenario_logistik as SL
from jst_shifting_bottleneck import shifting_bottleneck


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    seed: int = C.DEFAULT_SEED
    vehicle: str = C.DEFAULT_VEHICLE
    setup_time: int = C.DEFAULT_SETUP_TIME
    n_families: int = C.DEFAULT_N_FAMILIES
    max_iters: int = C.DEFAULT_MAX_ITERS
    tenure: int = C.DEFAULT_TENURE


@lru_cache(maxsize=512)
def instance(n, m, seed):
    return S.generate(n, m, seed)


@lru_cache(maxsize=512)
def logistik_instance(n, m, seed, n_families, setup_time):
    return SL.generate(n, m, seed, n_families=n_families, setup_time=setup_time)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    tabu: object                # Hauptergebnis: Tabu Search ab MWKR-Start
    mwkr: object                 # die Startlösung selbst (Stück 8) - Vergleichsbasis
    optimal: object              # None, wenn n > EXACT_MAX_N
    optimal_proven: bool

    @property
    def gap_mwkr(self):
        return _gap(self.mwkr.cmax, self.tabu.cmax)

    @property
    def tabu_matches_optimum(self):
        return self.optimal is not None and abs(self.tabu.cmax - self.optimal.cmax) < 1e-6

    @property
    def tabu_ratio_to_optimum(self):
        return None if self.optimal is None or self.optimal.cmax <= 1e-9 else self.tabu.cmax / self.optimal.cmax


def _gap(value, baseline):
    if baseline <= 1e-9:
        return 0.0 if value <= 1e-9 else float(value)
    return 100.0 * (value - baseline) / baseline


def analyse(settings):
    """Wertet Tabu Search (Start: MWKR/Giffler-Thompson, Stück 8) auf dem gewählten Vehikel aus."""
    if settings.vehicle == "logistik":
        inst = logistik_instance(settings.n, settings.m, settings.seed, settings.n_families, settings.setup_time)
        routing, proc, family, setup = inst.routing, inst.proc, inst.family, inst.setup
        optimal, proven = (A.solve_exact(routing, proc, family, setup, C.EXACT_TIME_LIMIT_SECONDS)
                            if settings.n <= C.EXACT_MAX_N else (None, False))
        mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc), family, setup)
        tabu = A.tabu_search(routing, proc, mwkr.start, mwkr.end, settings.max_iters, settings.tenure, family, setup, record_trace=True)
    else:
        inst = instance(settings.n, settings.m, settings.seed)
        routing, proc = inst.routing, inst.proc
        optimal, proven = (A.solve_exact(routing, proc, time_limit_seconds=C.EXACT_TIME_LIMIT_SECONDS)
                            if settings.n <= C.EXACT_MAX_N else (None, False))
        mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
        tabu = A.tabu_search(routing, proc, mwkr.start, mwkr.end, settings.max_iters, settings.tenure, record_trace=True)

    return Analysis(settings, inst, tabu, mwkr, optimal, proven)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        a = analyse(replace(s0, seed=seed))
        rows.append({"gap_mwkr": a.gap_mwkr})
    out = {k: _mean(rows, k) for k in rows[0]}
    out["n_runs"] = len(rows)
    return out


SWEEP_VALUES = {"n": (2, 4, 6, 8, 12, 16, 20), "m": (2, 3, 4, 5, 6)}
SWEEP_LABELS = {"n": "Aufträge", "m": "Maschinen"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def start_comparison(ns=(8, 10, 12, 15, 20), m=C.DEFAULT_M, seeds=C.SWEEP_SEEDS, max_iters=C.DEFAULT_MAX_ITERS, tenure=C.DEFAULT_TENURE):
    """Das zentrale Experiment dieses Stücks: verbessert ein besserer Start (Shifting Bottleneck, Stück 9)
    gegenüber dem naiven MWKR-Start (Stück 8) das Ergebnis von Tabu Search bei fester Budget-Grenze? Gemessen,
    nicht angenommen (siehe README für den überraschenden Befund: bei den meisten Instanzen praktisch kein
    Unterschied - anders als bei der Schwester-Tabu-Demo im Trajektorien-Metaheuristiken-Portfolio)."""
    rows = []
    for n in ns:
        from_mwkr, from_sb, ties = [], [], 0
        for seed in seeds:
            inst = instance(n, m, seed)
            mwkr = A.giffler_thompson(inst.routing, inst.proc, A.mwkr_priority(inst.proc))
            sb = shifting_bottleneck(inst.routing, inst.proc, reoptimize=True)
            tabu_mwkr = A.tabu_search(inst.routing, inst.proc, mwkr.start, mwkr.end, max_iters, tenure)
            tabu_sb = A.tabu_search(inst.routing, inst.proc, sb.start, sb.end, max_iters, tenure)
            from_mwkr.append(tabu_mwkr.cmax)
            from_sb.append(tabu_sb.cmax)
            if abs(tabu_mwkr.cmax - tabu_sb.cmax) < 1e-6:
                ties += 1
        rows.append({"value": n, "from_mwkr": float(np.mean(from_mwkr)), "from_sb": float(np.mean(from_sb)),
                      "tie_rate": ties / len(seeds)})
    return rows


def timing_sweep(ns=(2, 3, 4, 5, 6, 7, 8), m=C.DEFAULT_M, seed=C.DEFAULT_SEED):
    """Gemessene Rechenzeit: CP-SAT auf das VOLLE Job-Shop-Modell (im schlimmsten Fall exponentiell) gegen Tabu
    Search (reine Graph-/Kombinatorik-Operationen, kein CP-SAT im Hauptfluss)."""
    rows = []
    for n in ns:
        inst = instance(n, m, seed)
        t0 = time.perf_counter()
        A.solve_exact(inst.routing, inst.proc, time_limit_seconds=C.EXACT_TIME_LIMIT_SECONDS)
        t_exact = time.perf_counter() - t0
        mwkr = A.giffler_thompson(inst.routing, inst.proc, A.mwkr_priority(inst.proc))
        t0 = time.perf_counter()
        A.tabu_search(inst.routing, inst.proc, mwkr.start, mwkr.end, C.DEFAULT_MAX_ITERS, C.DEFAULT_TENURE)
        t_tabu = time.perf_counter() - t0
        rows.append({"value": n, "exact_seconds": t_exact, "tabu_seconds": t_tabu})
    return rows


def setup_gap(n=6, m=C.DEFAULT_M, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME):
    """Vehikel-B-Härtetest: Tabu Search (Start MWKR, rüstzeitbewusster kritischer Pfad) gegen die echte
    Optimallösung MIT Rüstzeiten (CP-SAT, deshalb kleines n)."""
    gaps = []
    for seed in seeds:
        linst = SL.generate(n, m, seed, n_families=n_families, setup_time=setup_time)
        mwkr = A.giffler_thompson(linst.routing, linst.proc, A.mwkr_priority(linst.proc), linst.family, linst.setup)
        tabu = A.tabu_search(linst.routing, linst.proc, mwkr.start, mwkr.end, C.DEFAULT_MAX_ITERS, C.DEFAULT_TENURE,
                              linst.family, linst.setup)
        opt, proven = A.solve_exact(linst.routing, linst.proc, linst.family, linst.setup, C.EXACT_TIME_LIMIT_SECONDS)
        if proven:
            gaps.append(_gap(tabu.cmax, opt.cmax))
    return {"gap_mean": float(np.mean(gaps)), "gap_min": float(np.min(gaps)), "gap_max": float(np.max(gaps)), "n_runs": len(gaps)}


def setup_gap_sweep(setup_times=(0, 5, 15, 30, 60), n=6, m=C.DEFAULT_M, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES):
    return [{"value": s, **setup_gap(n=n, m=m, seeds=seeds, n_families=n_families, setup_time=s)} for s in setup_times]
