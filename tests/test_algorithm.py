"""jst_algorithm: CP-SAT gegen unabhängige Brute-Force-Vollaufzählung (nur kleine n), kritischer Pfad und Blöcke
auf einer Handrechnung, die N5-Ausnahmeregel, der echte Fund beim Bau (alle Züge tabu darf die Suche NICHT
einfrieren lassen), Tabu Search gegen CP-SAT-Optimum, Rüstzeit-Variante."""

import itertools

import numpy as np
import pytest

import jst_algorithm as A


def _instance(seed, n, m):
    rng = np.random.default_rng(seed)
    routing = np.array([rng.permutation(m) for _ in range(n)])
    proc = rng.integers(1, 30, size=(n, m)).astype(np.int64)
    return routing, proc


def _brute_force(routing, proc, family=None, setup=None):
    """Alle Permutationen der Operationen je Maschine, nur zulässige (azyklische) Kombinationen ausgewertet."""
    n, m = routing.shape
    ops_per_machine = [[] for _ in range(m)]
    for j in range(n):
        for pos in range(m):
            ops_per_machine[routing[j, pos]].append((j, pos))
    best = None
    for combo in itertools.product(*(list(itertools.permutations(ops)) for ops in ops_per_machine)):
        machine_order = {k: list(combo[k]) for k in range(m)}
        machine_ptr = {k: 0 for k in range(m)}
        job_ptr = {j: 0 for j in range(n)}
        machine_free = [0] * m
        job_free = [0] * n
        prev_family = [None] * m
        end_time = {}
        scheduled, total = 0, n * m
        progress = True
        while scheduled < total and progress:
            progress = False
            for k in range(m):
                if machine_ptr[k] >= len(machine_order[k]):
                    continue
                j, pos = machine_order[k][machine_ptr[k]]
                if job_ptr[j] == pos:
                    s = int(setup[prev_family[k], family[j]]) if family is not None and prev_family[k] is not None else 0
                    st = max(machine_free[k] + s, job_free[j])
                    dur = int(proc[j, pos])
                    end_time[j, pos] = st + dur
                    machine_free[k] = st + dur
                    job_free[j] = st + dur
                    if family is not None:
                        prev_family[k] = family[j]
                    machine_ptr[k] += 1
                    job_ptr[j] += 1
                    scheduled += 1
                    progress = True
        if scheduled < total:
            continue
        cmax = max(end_time.values())
        if best is None or cmax < best:
            best = cmax
    return best


@pytest.mark.parametrize("n,m", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_cp_sat_matches_brute_force_for_every_seed(n, m):
    for seed in range(5):
        routing, proc = _instance(seed * 10 + n + m, n, m)
        bf = _brute_force(routing, proc)
        result, proven = A.solve_exact(routing, proc, time_limit_seconds=10)
        assert proven
        assert result.cmax == pytest.approx(bf, abs=1e-6)


def test_giffler_thompson_produces_a_feasible_precedence_respecting_schedule():
    routing, proc = _instance(5, 6, 3)
    result = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    n, m = routing.shape
    for j in range(n):
        for pos in range(1, m):
            assert result.start[j, pos] >= result.end[j, pos - 1] - 1e-9
    for k in range(m):
        ops_k = sorted([(j, pos) for j in range(n) for pos in range(m) if routing[j, pos] == k], key=lambda jp: result.start[jp])
        for a, b in zip(ops_k, ops_k[1:]):
            assert result.start[b] >= result.end[a] - 1e-9


def test_schedule_from_machine_seq_roundtrips_a_giffler_thompson_schedule():
    """Liest man die Maschinenreihenfolge eines Zeitplans ab und rechnet ihn daraus zurück, muss GENAU derselbe
    Zeitplan herauskommen (beide Berechnungswege sind längste-Wege-Relaxationen desselben Graphen)."""
    routing, proc = _instance(7, 8, 4)
    result = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    machine_seq = A.machine_seq_from_schedule(routing, result.start)
    s2, e2 = A.schedule_from_machine_seq(routing, proc, machine_seq)
    assert s2 is not None
    assert np.allclose(s2, result.start, atol=1e-6)
    assert np.allclose(e2, result.end, atol=1e-6)


# --- Kritischer Pfad + Blöcke: Handrechnung --------------------------------------------------------------------


def test_critical_path_on_a_hand_picked_two_job_two_machine_instance():
    """2 Aufträge, 2 Maschinen, gegenläufige Reihenfolgen: Auftrag 0 läuft M0->M1, Auftrag 1 läuft M1->M0. Bei
    fester Priorität (Auftrag 0 zuerst auf jeder Maschine) ist der kritische Pfad von Hand nachvollziehbar."""
    routing = np.array([[0, 1], [1, 0]])
    proc = np.array([[5, 3], [4, 6]])
    result = A.giffler_thompson(routing, proc, lambda j, pos: j)
    path = A.critical_path(routing, proc, result.start, result.end)
    # jede Operation im Pfad muss straff an ihren Nachfolger anschließen
    for a, b in zip(path, path[1:]):
        assert result.end[a[0], a[1]] == pytest.approx(result.start[b[0], b[1]], abs=1e-6)
    # der Pfad endet an der Operation mit Cmax
    assert result.end[path[-1][0], path[-1][1]] == pytest.approx(result.cmax, abs=1e-6)


def test_blocks_from_critical_path_groups_consecutive_same_machine_operations():
    path = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    routing = np.zeros((2, 3), dtype=int)
    routing[0, 0], routing[0, 1] = 5, 5   # (0,0),(0,1) beide Maschine 5
    routing[1, 0], routing[1, 1], routing[1, 2] = 5, 2, 2
    blocks = A.blocks_from_critical_path(routing, path)
    assert blocks == [[(0, 0), (0, 1), (1, 0)], [(1, 1), (1, 2)]]


def test_n5_neighbors_excludes_first_pair_of_first_block_and_last_pair_of_last_block():
    """Nowicki & Smutnicki (1996): am ersten Block wird das ERSTE Paar NICHT getauscht, am letzten Block das
    LETZTE Paar NICHT - sonst je EIN Zug pro Block mit >=2 Operationen."""
    routing = np.array([[0], [0], [0], [1], [1], [1]])  # Jobs 0-2 -> Maschine 0, Jobs 3-5 -> Maschine 1
    machine_seq = {0: [(0, 0), (1, 0), (2, 0)], 1: [(3, 0), (4, 0), (5, 0)]}
    blocks = [[(0, 0), (1, 0), (2, 0)], [(3, 0), (4, 0), (5, 0)]]
    moves = A.n5_neighbors(routing, machine_seq, blocks)
    # Block 0 (erster UND nicht letzter): nur "letztes Paar" erlaubt -> Tausch von (1,0)/(2,0)
    # Block 1 (letzter UND nicht erster): nur "erstes Paar" erlaubt -> Tausch von (3,0)/(4,0)
    expected = [(0, [(0, 0), (2, 0), (1, 0)]), (1, [(4, 0), (3, 0), (5, 0)])]
    assert moves == expected


def test_n5_neighbors_offers_both_pairs_for_an_interior_block():
    routing = np.zeros((10, 1), dtype=int)
    routing[0:4, 0] = 5   # Jobs 0-3 -> Maschine 5 (Jobs 8/9 sind Ein-Operationen-Blöcke, Maschine egal)
    machine_seq = {5: [(0, 0), (1, 0), (2, 0), (3, 0)]}
    blocks = [[(9, 0)], [(0, 0), (1, 0), (2, 0), (3, 0)], [(8, 0)]]  # Block 1 ist WEDER erster NOCH letzter
    moves = A.n5_neighbors(routing, machine_seq, blocks)
    assert len(moves) == 2   # sowohl erstes als auch letztes Paar erlaubt


def test_n5_neighbors_offers_no_move_when_the_only_block_has_one_operation():
    routing = np.array([[5]])
    machine_seq = {5: [(0, 0)]}
    blocks = [[(0, 0)]]
    assert A.n5_neighbors(routing, machine_seq, blocks) == []


# --- Tabu Search ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,m", [(2, 2), (3, 2), (3, 3), (4, 3)])
def test_tabu_search_never_beats_the_cp_sat_optimum(n, m):
    for seed in range(3):
        routing, proc = _instance(seed * 10 + n + m, n, m)
        mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
        result = A.tabu_search(routing, proc, mwkr.start, mwkr.end, max_iters=100, tenure=6)
        opt, proven = A.solve_exact(routing, proc, time_limit_seconds=10)
        assert proven
        assert result.cmax >= opt.cmax - 1e-6


def test_tabu_search_never_makes_the_start_worse():
    """Die BESTE je gefundene Lösung wird zurückgegeben - auch wenn Tabu Search zwischendurch verschlechternde
    Züge macht, kann das Endergebnis nie schlechter als die Startlösung selbst sein."""
    routing, proc = _instance(9, 10, 4)
    mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    result = A.tabu_search(routing, proc, mwkr.start, mwkr.end, max_iters=200, tenure=8)
    assert result.cmax <= mwkr.cmax + 1e-6


def test_tabu_search_produces_a_feasible_precedence_respecting_schedule():
    routing, proc = _instance(9, 10, 4)
    mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    result = A.tabu_search(routing, proc, mwkr.start, mwkr.end, max_iters=200, tenure=8)
    n, m = routing.shape
    for j in range(n):
        for pos in range(1, m):
            assert result.start[j, pos] >= result.end[j, pos - 1] - 1e-6
    for k in range(m):
        ops_k = sorted([(j, pos) for j in range(n) for pos in range(m) if routing[j, pos] == k], key=lambda jp: result.start[jp])
        for a, b in zip(ops_k, ops_k[1:]):
            assert result.start[b] >= result.end[a] - 1e-6


def test_all_tabu_moves_fall_back_instead_of_freezing_the_search():
    """Regressionsschutz für einen echten, gefundenen Fund (n=6, m=3, Seed 3): der kritische Pfad hat nur EINEN
    möglichen Zug, der nach einer Runde exakt rückgängig gemacht würde (tabu) - ohne den Rückfall auf den am
    wenigsten schlechten tabu-Zug fror die Suche hier bei der Startlösung ein (0 % Verbesserung über 1000
    Iterationen jeder Tenure). Mit dem Fix erreicht diese Instanz sogar das CP-SAT-Optimum."""
    rng = np.random.default_rng(3)
    routing = np.array([rng.permutation(3) for _ in range(6)])
    proc = rng.integers(1, 30, size=(6, 3)).astype(np.int64)
    mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    result = A.tabu_search(routing, proc, mwkr.start, mwkr.end, max_iters=100, tenure=6)
    assert result.cmax < mwkr.cmax - 1e-6


def test_record_trace_starts_at_the_initial_solution_and_ends_at_the_best():
    routing, proc = _instance(4, 8, 4)
    mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    result = A.tabu_search(routing, proc, mwkr.start, mwkr.end, max_iters=200, tenure=8, record_trace=True)
    assert result.trace[0]["cmax"] == pytest.approx(mwkr.cmax)
    assert result.trace[-1]["cmax"] == pytest.approx(result.cmax)
    assert all(result.trace[i]["cmax"] >= result.trace[i + 1]["cmax"] - 1e-9 for i in range(len(result.trace) - 1))


def test_single_critical_block_offers_zero_moves():
    """Dokumentierter Grenzfall der N5-Nachbarschaft (siehe README/🚧): liegt der GESAMTE kritische Pfad auf
    einer einzigen Maschine, greift die Ausnahmeregel doppelt (erster UND letzter Block zugleich) - keine
    Verbesserung von dieser Startlösung aus möglich."""
    routing = np.array([[0], [0], [0]])
    proc = np.array([[5], [3], [7]])
    mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    result = A.tabu_search(routing, proc, mwkr.start, mwkr.end, max_iters=100, tenure=6)
    assert result.cmax == pytest.approx(mwkr.cmax)


# --- Mit Rüstzeiten (Vehikel B) --------------------------------------------------------------------------------


def test_tabu_search_at_zero_setup_time_matches_the_neutral_vehicle_exactly():
    routing, proc = _instance(7, 6, 3)
    rng = np.random.default_rng(1)
    family = rng.integers(0, 3, size=6)
    setup = np.zeros((3, 3))
    mwkr_n = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    neutral = A.tabu_search(routing, proc, mwkr_n.start, mwkr_n.end, max_iters=100, tenure=6)
    mwkr_l = A.giffler_thompson(routing, proc, A.mwkr_priority(proc), family, setup)
    logistik = A.tabu_search(routing, proc, mwkr_l.start, mwkr_l.end, max_iters=100, tenure=6, family=family, setup=setup)
    assert neutral.cmax == pytest.approx(logistik.cmax)


def test_cp_sat_with_setup_matches_independent_brute_force():
    routing, proc = _instance(11, 3, 3)
    rng = np.random.default_rng(11)
    family = rng.integers(0, 2, size=3)
    setup = np.array([[0, 6], [6, 0]])
    bf = _brute_force(routing, proc, family, setup)
    result, proven = A.solve_exact(routing, proc, family, setup, time_limit_seconds=10)
    assert proven
    assert result.cmax == pytest.approx(bf, abs=1e-6)


def test_tabu_search_with_setup_never_beats_cp_sat_optimum():
    routing, proc = _instance(20, 5, 3)
    rng = np.random.default_rng(2)
    family = rng.integers(0, 3, size=5)
    setup = np.full((3, 3), 15)
    np.fill_diagonal(setup, 0)
    mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc), family, setup)
    result = A.tabu_search(routing, proc, mwkr.start, mwkr.end, max_iters=100, tenure=6, family=family, setup=setup)
    opt, proven = A.solve_exact(routing, proc, family, setup, time_limit_seconds=10)
    assert proven
    assert result.cmax >= opt.cmax - 1e-6
