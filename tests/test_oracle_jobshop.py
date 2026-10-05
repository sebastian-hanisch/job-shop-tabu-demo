"""Orakel-Tests (anderer Rechenweg): Vollaufzählung aller Maschinenreihenfolgen mit eigener Längster-Weg-Rechnung
(rekursiv, mit Zyklenerkennung) statt CP-SAT, Kahn-Iteration und Tabu-Schleife der Demo. Klein und schnell (< 10 s)."""
import itertools
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jst_algorithm as A  # noqa: E402
import jst_scenario as S  # noqa: E402
import jst_scenario_logistik as SL  # noqa: E402
from jst_shifting_bottleneck import shifting_bottleneck  # noqa: E402


def setup_of(family, setup, a, b):
    return 0 if family is None else int(setup[family[a[0]], family[b[0]]])


def indep_schedule(routing, proc, seq, family=None, setup=None):
    """Semi-aktiver Zeitplan (Startzeiten) per rekursiver Auswertung; None bei Zyklus."""
    n, m = routing.shape
    mpred = {s[i]: s[i - 1] for s in seq.values() for i in range(1, len(s))}
    start, state = {}, {}

    def go(op):
        if state.get(op) == 2:
            return start[op]
        if state.get(op) == 1:
            raise ValueError("Zyklus")
        state[op] = 1
        j, p = op
        v = 0
        if p > 0:
            q = (j, p - 1)
            v = max(v, go(q) + proc[q])
        if op in mpred:
            q = mpred[op]
            v = max(v, go(q) + proc[q] + setup_of(family, setup, q, op))
        start[op], state[op] = v, 2
        return v

    try:
        for j in range(n):
            for p in range(m):
                go((j, p))
    except ValueError:
        return None
    arr = np.zeros((n, m))
    for (j, p), v in start.items():
        arr[j, p] = v
    return arr


def brute_opt(routing, proc, family=None, setup=None):
    n, m = routing.shape
    ops = [[(j, p) for j in range(n) for p in range(m) if routing[j, p] == k] for k in range(m)]
    best = None
    for perms in itertools.product(*[list(itertools.permutations(o)) for o in ops]):
        st = indep_schedule(routing, proc, {k: list(perms[k]) for k in range(m)}, family, setup)
        if st is not None:
            c = (st + proc).max()
            best = c if best is None or c < best else best
    return best


def validate(res, routing, proc, family=None, setup=None):
    n, m = routing.shape
    s, e = res.start, res.end
    assert np.allclose(e, s + proc) and (s >= -1e-9).all()
    for j in range(n):
        for p in range(1, m):
            assert s[j, p] >= e[j, p - 1] - 1e-9
    for k in range(m):
        ops = sorted([(j, p) for j in range(n) for p in range(m) if routing[j, p] == k], key=lambda o: s[o])
        for a, b in zip(ops, ops[1:]):
            assert s[b] >= e[a] + setup_of(family, setup, a, b) - 1e-9
    assert abs(res.cmax - e.max()) < 1e-9


def instances(rng, count, logistik):
    for _ in range(count):
        n, m = rng.choice([(2, 3), (3, 2), (3, 3), (4, 2)])
        seed = rng.randrange(10 ** 6)
        if logistik:
            i = SL.generate(n, m, seed, n_families=3, setup_time=rng.choice([3, 15, 60]), p_max=rng.choice([10, 100]))
            yield i.routing, i.proc, i.family, i.setup
        else:
            i = S.generate(n, m, seed, p_max=rng.choice([10, 100]))
            yield i.routing, i.proc, None, None


@pytest.mark.parametrize("logistik", [False, True])
def test_all_methods_against_full_enumeration(logistik):
    rng = random.Random(5 + logistik)
    for routing, proc, family, setup in instances(rng, 8, logistik):
        opt = brute_opt(routing, proc, family, setup)
        ex, proven = A.solve_exact(routing, proc, family, setup, 20.0)
        assert proven and ex.cmax == pytest.approx(opt)
        validate(ex, routing, proc, family, setup)
        gt = A.giffler_thompson(routing, proc, A.mwkr_priority(proc), family, setup)
        validate(gt, routing, proc, family, setup)
        tb = A.tabu_search(routing, proc, gt.start, gt.end, 100, 5, family, setup)
        validate(tb, routing, proc, family, setup)
        assert opt - 1e-9 <= tb.cmax <= gt.cmax + 1e-9 and gt.cmax >= opt - 1e-9


def test_schedule_from_machine_seq_matches_independent_longest_path_and_cycles():
    rng = random.Random(8)
    for t in range(150):
        n, m = rng.randint(2, 6), rng.randint(2, 4)
        i = SL.generate(n, m, rng.randrange(10 ** 6), n_families=3, setup_time=15) if t % 2 else S.generate(n, m, rng.randrange(10 ** 6))
        family, setup = (i.family, i.setup) if t % 2 else (None, None)
        seq = {}
        for k in range(m):
            ops = [(j, p) for j in range(n) for p in range(m) if i.routing[j, p] == k]
            rng.shuffle(ops)
            seq[k] = ops
        ref = indep_schedule(i.routing, i.proc, seq, family, setup)
        s, e = A.schedule_from_machine_seq(i.routing, i.proc, seq, family, setup)
        if ref is None:
            assert s is None
        else:
            assert np.allclose(s, ref) and np.allclose(e, ref + i.proc)


def _excluded_end_moves(routing, mseq, blocks):
    cand = []
    if len(blocks[0]) >= (2 if len(blocks) == 1 else 3):
        k = routing[blocks[0][0]]
        s2 = list(mseq[k])
        i0 = s2.index(blocks[0][0])
        s2[i0], s2[i0 + 1] = s2[i0 + 1], s2[i0]
        cand.append((k, s2))
    if len(blocks[-1]) >= (2 if len(blocks) == 1 else 3):
        k = routing[blocks[-1][0]]
        s2 = list(mseq[k])
        i0 = s2.index(blocks[-1][0]) + len(blocks[-1]) - 2
        s2[i0], s2[i0 + 1] = s2[i0 + 1], s2[i0]
        cand.append((k, s2))
    return cand


def test_n5_excluded_end_pairs_never_improve_without_setups():
    """Satz von Nowicki & Smutnicki (ohne Rüstzeiten): die ausgeschlossenen Züge verkleinern Cmax nie."""
    rng = random.Random(9)
    checked = 0
    for _ in range(120):
        n, m = rng.randint(3, 8), rng.randint(2, 5)
        i = S.generate(n, m, rng.randrange(10 ** 6))
        gt = A.giffler_thompson(i.routing, i.proc, A.mwkr_priority(i.proc))
        path = A.critical_path(i.routing, i.proc, gt.start, gt.end)
        blocks = A.blocks_from_critical_path(i.routing, path)
        mseq = A.machine_seq_from_schedule(i.routing, gt.start)
        for k, s2 in _excluded_end_moves(i.routing, mseq, blocks):
            tr = dict(mseq)
            tr[k] = s2
            st = indep_schedule(i.routing, i.proc, tr)
            if st is not None:
                checked += 1
                assert (st + i.proc).max() >= gt.cmax - 1e-9
    assert checked > 20


def test_n5_end_pairs_are_offered_with_setup_times():
    """Regression (Orakel-Fund): mit Rüstzeiten kann ein Tausch am äußersten Paar Cmax verkleinern; die Ausnahme
    ließ Tabu Search hier bei 212 stehen (Optimum 197). Einzelner kritischer Block: N5 bot keinen Zug an."""
    l = SL.generate(3, 2, 63, n_families=3, setup_time=15)
    gt = A.giffler_thompson(l.routing, l.proc, A.mwkr_priority(l.proc), l.family, l.setup)
    path = A.critical_path(l.routing, l.proc, gt.start, gt.end, l.family, l.setup)
    blocks = A.blocks_from_critical_path(l.routing, path)
    mseq = A.machine_seq_from_schedule(l.routing, gt.start)
    improving = []
    for k, s2 in _excluded_end_moves(l.routing, mseq, blocks):
        tr = dict(mseq)
        tr[k] = s2
        st = indep_schedule(l.routing, l.proc, tr, l.family, l.setup)
        if st is not None and (st + l.proc).max() < gt.cmax - 1e-9:
            improving.append((k, s2))
    assert improving, "Testinstanz muss einen verbessernden Randzug haben"
    assert all(mv in A.n5_neighbors(l.routing, mseq, blocks, exclude_ends=False) for mv in improving)
    tb = A.tabu_search(l.routing, l.proc, gt.start, gt.end, 200, 8, l.family, l.setup)
    opt = brute_opt(l.routing, l.proc, l.family, l.setup)
    assert tb.cmax < gt.cmax - 1e-9 and tb.cmax == pytest.approx(opt)


def test_shifting_bottleneck_is_feasible_and_not_below_optimum():
    rng = random.Random(11)
    for logistik in (False, True):
        for routing, proc, family, setup in instances(rng, 3, logistik):
            opt = brute_opt(routing, proc, family, setup)
            sb = shifting_bottleneck(routing, proc, family, setup, reoptimize=True)
            validate(sb, routing, proc, family, setup)
            assert sb.cmax >= opt - 1e-9
