"""Shifting Bottleneck (Adams, Balas & Zawack 1988) als ZWEITE Startlösung für die Tabu-Suche - kopiert (nicht
importiert, siehe `sb_algorithm.py` in `shifting-bottleneck-demo`, Stück 9 dieser Linie) für die Messfrage dieses
Stücks: hilft ein besserer Start (Shifting Bottleneck) gegenüber dem naiven MWKR-Start (Stück 8) bei fester
Budget-Grenze? Jedes Stück dieser Linie ist eigenständig lauffähig, deshalb Kopie statt Cross-Repo-Import -
derselbe Grundsatz wie `johnson_algorithm.py`s MWKR-Kopie aus Stück 8 oder `gls_tabu_algorithm.py`s Tabu-Kopie
im Trajektorien-Metaheuristiken-Portfolio."""

import math

import numpy as np
from ortools.sat.python import cp_model

from jst_algorithm import NUM_SEARCH_WORKERS, Result


def compute_heads(routing, proc, machine_seq, family=None, setup=None):
    n, m = routing.shape
    head = np.zeros((n, m))
    pred_on_machine = {}
    pred_setup = {}
    for seq in machine_seq.values():
        for idx in range(1, len(seq)):
            pred_on_machine[seq[idx]] = seq[idx - 1]
            if family is not None:
                pj, pp = seq[idx - 1]
                j2, p2 = seq[idx]
                pred_setup[seq[idx]] = int(setup[family[pj], family[j2]])
    for _ in range(n * m + 2):
        changed = False
        for j in range(n):
            for pos in range(m):
                val = 0.0
                if pos > 0:
                    val = max(val, head[j, pos - 1] + proc[j, pos - 1])
                pred = pred_on_machine.get((j, pos))
                if pred is not None:
                    pj, pp = pred
                    s = pred_setup.get((j, pos), 0)
                    val = max(val, head[pj, pp] + proc[pj, pp] + s)
                if val != head[j, pos]:
                    head[j, pos] = val
                    changed = True
        if not changed:
            break
    return head


def compute_tails(routing, proc, machine_seq, family=None, setup=None):
    n, m = routing.shape
    tail = np.zeros((n, m))
    succ_on_machine = {}
    succ_setup = {}
    for seq in machine_seq.values():
        for idx in range(len(seq) - 1):
            succ_on_machine[seq[idx]] = seq[idx + 1]
            if family is not None:
                j1, p1 = seq[idx]
                sj, sp = seq[idx + 1]
                succ_setup[seq[idx]] = int(setup[family[j1], family[sj]])
    for _ in range(n * m + 2):
        changed = False
        for j in range(n):
            for pos in range(m - 1, -1, -1):
                val = 0.0
                if pos < m - 1:
                    val = max(val, tail[j, pos + 1] + proc[j, pos + 1])
                succ = succ_on_machine.get((j, pos))
                if succ is not None:
                    sj, sp = succ
                    s = succ_setup.get((j, pos), 0)
                    val = max(val, tail[sj, sp] + proc[sj, sp] + s)
                if val != tail[j, pos]:
                    tail[j, pos] = val
                    changed = True
        if not changed:
            break
    return tail


def solve_1r_lmax(ops, proc, head, tail, horizon, family=None, setup=None, time_limit_seconds=5.0):
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1  # winziges, oft gelöstes Teilproblem - Determinismus über Umgebungen
    solver.parameters.max_time_in_seconds = time_limit_seconds

    if family is None or int(np.max(setup)) == 0:
        starts, ends, intervals = {}, {}, []
        for (j, pos) in ops:
            r, d = int(head[j, pos]), int(proc[j, pos])
            s = model.NewIntVar(r, horizon, f"s{j}_{pos}")
            e = model.NewIntVar(r, horizon, f"e{j}_{pos}")
            model.Add(e == s + d)
            starts[j, pos], ends[j, pos] = s, e
            intervals.append(model.NewIntervalVar(s, d, e, f"iv{j}_{pos}"))
        model.AddNoOverlap(intervals)
    else:
        starts, ends = {}, {}
        for (j, pos) in ops:
            r, d = int(head[j, pos]), int(proc[j, pos])
            starts[j, pos] = model.NewIntVar(r, horizon, f"s{j}_{pos}")
            ends[j, pos] = model.NewIntVar(r, horizon, f"e{j}_{pos}")
            model.Add(ends[j, pos] == starts[j, pos] + d)
        arcs = []
        for idx, (j, pos) in enumerate(ops):
            arcs.append((0, idx + 1, model.NewBoolVar(f"a0_{idx}")))
            arcs.append((idx + 1, 0, model.NewBoolVar(f"a{idx}_0")))
        for i1, (j1, p1) in enumerate(ops):
            for i2, (j2, p2) in enumerate(ops):
                if i1 == i2:
                    continue
                lit = model.NewBoolVar(f"a{i1}_{i2}")
                arcs.append((i1 + 1, i2 + 1, lit))
                s = int(setup[family[j1], family[j2]])
                model.Add(starts[j2, p2] >= ends[j1, p1] + s).OnlyEnforceIf(lit)
        model.AddCircuit(arcs)

    lmax = model.NewIntVar(-horizon, horizon, "lmax")
    for (j, pos) in ops:
        due = horizon - int(tail[j, pos])
        model.Add(lmax >= ends[j, pos] - due)
    model.Minimize(lmax)
    solver.Solve(model)
    seq = sorted(ops, key=lambda jp: solver.Value(starts[jp]))
    return seq, solver.Value(lmax)


def _is_acyclic(routing, machine_seq):
    n, m = routing.shape
    indeg = {(j, pos): 0 for j in range(n) for pos in range(m)}
    adj = {(j, pos): [] for j in range(n) for pos in range(m)}

    def add_edge(u, v):
        adj[u].append(v)
        indeg[v] += 1

    for j in range(n):
        for pos in range(1, m):
            add_edge((j, pos - 1), (j, pos))
    for seq in machine_seq.values():
        for idx in range(1, len(seq)):
            add_edge(seq[idx - 1], seq[idx])

    queue = [node for node, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return visited == n * m


def _assign_machine_sequence(routing, machine_seq, k, candidate_seq, head):
    trial = dict(machine_seq)
    trial[k] = candidate_seq
    if _is_acyclic(routing, trial):
        return candidate_seq
    return sorted(candidate_seq, key=lambda jp: head[jp[0], jp[1]])


def shifting_bottleneck(routing, proc, family=None, setup=None, reoptimize=True, time_limit_seconds=5.0):
    """Adams, Balas & Zawack (1988) - Kurzfassung (siehe `shifting-bottleneck-demo`/Stück 9 für die volle
    Herleitung inkl. Zyklen-Schutz und Nichtdeterminismus-Fix, hier unverändert übernommen)."""
    n, m = routing.shape
    horizon = int(proc.sum()) + (int(setup.max()) * n * m if setup is not None else 0) + 1
    ops_per_machine = [[] for _ in range(m)]
    for j in range(n):
        for pos in range(m):
            ops_per_machine[routing[j, pos]].append((j, pos))

    machine_seq = {}
    unsequenced = set(range(m))
    while unsequenced:
        head = compute_heads(routing, proc, machine_seq, family, setup)
        tail = compute_tails(routing, proc, machine_seq, family, setup)
        best_k, best_seq, best_lmax = None, None, -math.inf
        for k in unsequenced:
            seq, lmax = solve_1r_lmax(ops_per_machine[k], proc, head, tail, horizon, family, setup, time_limit_seconds)
            if lmax > best_lmax:
                best_k, best_seq, best_lmax = k, seq, lmax
        machine_seq[best_k] = _assign_machine_sequence(routing, machine_seq, best_k, best_seq, head)
        unsequenced.discard(best_k)
        if reoptimize:
            for k in list(machine_seq.keys()):
                if k == best_k:
                    continue
                trial = dict(machine_seq)
                del trial[k]
                head2 = compute_heads(routing, proc, trial, family, setup)
                tail2 = compute_tails(routing, proc, trial, family, setup)
                seq2, _ = solve_1r_lmax(ops_per_machine[k], proc, head2, tail2, horizon, family, setup, time_limit_seconds)
                machine_seq[k] = _assign_machine_sequence(routing, trial, k, seq2, head2)

    head_final = compute_heads(routing, proc, machine_seq, family, setup)
    start = head_final
    end = np.zeros((n, m))
    machine_finish = np.zeros(m)
    for j in range(n):
        for pos in range(m):
            end[j, pos] = start[j, pos] + proc[j, pos]
            machine_finish[routing[j, pos]] = max(machine_finish[routing[j, pos]], end[j, pos])
    return Result(start, end, float(end.max()), machine_finish)
