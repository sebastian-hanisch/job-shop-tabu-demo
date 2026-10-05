"""Job-Shop-Tabu-Search (Nowicki & Smutnicki 1996, Management Science 42(6):797-813) - das zweite SOTA-Stück
dieser Linie: statt einmal gut zu KONSTRUIEREN (Shifting Bottleneck, Stück 9) verbessert dieses Verfahren eine
gegebene Startlösung iterativ über eine gezielte, kleine Nachbarschaft.

Vier Bausteine:
1. **Kritischer Pfad**: der Weg vom ersten zum letzten Operationsende ohne jeden Puffer (jede Kante ist "straff":
   Nachfolger beginnt GENAU dann, wenn Vorgänger endet - plus Rüstzeit, falls Vehikel B).
2. **Blöcke**: der kritische Pfad wird in maximale Teilfolgen zerlegt, die auf DERSELBEN Maschine liegen -
   Operationen außerhalb eines Blocks liegen nicht auf dem kritischen Pfad und ihre Reihenfolge zu ändern kann
   Cmax nicht verbessern.
3. **N5-Nachbarschaft** (Nowicki & Smutnicki 1996, WebSearch-verifiziert): pro Block mit ≥2 Operationen wird NUR
   das ERSTE oder LETZTE Operationspaar vertauscht - mit einer Ausnahme aus der Originalarbeit: das erste Paar
   des ERSTEN Blocks und das letzte Paar des LETZTEN Blocks werden NICHT vertauscht (diese Züge können Cmax
   ohne Rüstzeiten beweisbar nicht verbessern; mit Rüstzeiten (Vehikel B) können sie es, deshalb entfällt die
   Ausnahme dort - Orakel-Fund, siehe README). Eine winzige, gezielte Nachbarschaft im Vergleich zur vollen paarweisen
   Nachbarschaft der Schwester-Demo `tabu-search-demo` (Trajektorien-Metaheuristiken-Linie, TSP).
4. **Tabu-Liste**: verhindert das sofortige Zurücktauschen eines gerade gemachten Zugs für `tenure` Iterationen,
   außer eine Aspiration (neues Gesamtoptimum) erlaubt es trotzdem. Sind ALLE Züge einer Iteration tabu, wird
   TROTZDEM der am wenigsten schlechte davon gemacht (Standard-Tabu-Search-Praxis) - ohne diesen Rückfall friert
   die Suche in kleinen Nachbarschaften (z. B. genau zwei sich gegenseitig aufhebende Blöcke) sofort ein, ein
   echter Fund beim Bau dieses Stücks (siehe README).

Giffler-Thompson/MWKR (Stück 8, hier als Startlösung kopiert, nicht importiert - jedes Stück dieser Linie ist
eigenständig lauffähig) und das volle CP-SAT-Kreis-Modell (exakte Gegenprobe für kleine n) sind unverändert. Reine
Graph-/Kombinatorik-Operationen - Tabu Search selbst nutzt KEIN CP-SAT, deshalb kein Nichtdeterminismus-Thema wie
in Stück 9."""

import os
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

NUM_SEARCH_WORKERS = min(8, os.cpu_count() or 1)  # NIE hart auf eine Zahl setzen - siehe project memory


@dataclass
class Result:
    start: np.ndarray          # (n, m): Startzeit je Operation
    end: np.ndarray            # (n, m): Fertigstellung je Operation
    cmax: float
    machine_finish: np.ndarray  # (m,): Fertigstellung der letzten Operation je Maschine
    trace: object = None        # nur tabu_search(record_trace=True): Liste von (it, cmax, start, end, critical_ops) bei jedem neuen Bestwert


# --- Giffler-Thompson/MWKR (Stück 8 dieser Linie, Startlösung) --------------------------------------------------


def _schedulable(routing, job_ptr):
    n, m = routing.shape
    return [(j, job_ptr[j]) for j in range(n) if job_ptr[j] < m]


def _setup_cost(family, setup, prev_family_on_machine, j):
    if family is None or prev_family_on_machine is None:
        return 0
    return int(setup[prev_family_on_machine, family[j]])


def giffler_thompson(routing, proc, priority, family=None, setup=None):
    """Konstruiert EINEN aktiven Zeitplan per Prioritätsregel - unverändert aus Stück 8, hier als Startlösung
    für die Tabu-Suche."""
    n, m = routing.shape
    job_ptr = [0] * n
    job_free = np.zeros(n)
    machine_free = np.zeros(m)
    prev_family_on_machine = [None] * m
    start = np.zeros((n, m))
    end = np.zeros((n, m))
    for _ in range(n * m):
        ops = _schedulable(routing, job_ptr)
        best = None
        for (j, pos) in ops:
            k = routing[j, pos]
            s = _setup_cost(family, setup, prev_family_on_machine[k], j)
            st = max(job_free[j], machine_free[k] + s)
            ct = st + proc[j, pos]
            if best is None or ct < best[0]:
                best = (ct, k)
        cstar, mstar = best
        conflict = []
        for (j, pos) in ops:
            k = routing[j, pos]
            if k != mstar:
                continue
            s = _setup_cost(family, setup, prev_family_on_machine[k], j)
            st = max(job_free[j], machine_free[k] + s)
            if st < cstar:
                conflict.append((j, pos))
        j, pos = min(conflict, key=lambda jp: priority(jp[0], jp[1]))
        k = routing[j, pos]
        s = _setup_cost(family, setup, prev_family_on_machine[k], j)
        st = max(job_free[j], machine_free[k] + s)
        start[j, pos], end[j, pos] = st, st + proc[j, pos]
        job_free[j] = end[j, pos]
        machine_free[k] = end[j, pos]
        if family is not None:
            prev_family_on_machine[k] = family[j]
        job_ptr[j] += 1
    machine_finish = np.zeros(m)
    for j in range(n):
        for pos in range(m):
            machine_finish[routing[j, pos]] = max(machine_finish[routing[j, pos]], end[j, pos])
    return Result(start, end, float(end.max()), machine_finish)


def mwkr_priority(proc):
    """Most Work Remaining - die beste einfache Regel aus Stück 8, hier die Startlösung für die Tabu-Suche."""
    def key(j, pos):
        return -float(proc[j, pos:].sum())
    return key


# --- Maschinenreihenfolge <-> Zeitplan --------------------------------------------------------------------------


def machine_seq_from_schedule(routing, start):
    """Liest die Maschinenreihenfolge aus einem gegebenen Zeitplan ab (Operationen je Maschine nach Startzeit
    sortiert)."""
    n, m = routing.shape
    seq = {k: [] for k in range(m)}
    ops = [(j, pos) for j in range(n) for pos in range(m)]
    ops.sort(key=lambda jp: start[jp[0], jp[1]])
    for (j, pos) in ops:
        seq[routing[j, pos]].append((j, pos))
    return seq


def _is_acyclic_and_longest_path(routing, proc, machine_seq, family, setup):
    """Kahn-Azyklizitätsprüfung PLUS längster-Weg-Berechnung in einem Durchgang (derselbe Zyklen-Schutz wie in
    `shifting-bottleneck-demo`, hier nötig, weil ein N5-Tausch zweier Operationen theoretisch einen Zyklus
    zwischen Auftragsvorrang und der neuen Maschinenreihenfolge schließen könnte). Gibt (None, None) zurück, wenn
    die kombinierte Reihenfolge nicht azyklisch ist."""
    n, m = routing.shape
    pred_on_machine = {}
    pred_setup = {}
    for seq in machine_seq.values():
        for idx in range(1, len(seq)):
            pred_on_machine[seq[idx]] = seq[idx - 1]
            if family is not None:
                pj, pp = seq[idx - 1]
                j2, p2 = seq[idx]
                pred_setup[seq[idx]] = int(setup[family[pj], family[j2]])

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
    if visited != n * m:
        return None, None

    start = np.zeros((n, m))
    for _ in range(n * m + 2):
        changed = False
        for j in range(n):
            for pos in range(m):
                val = 0.0
                if pos > 0:
                    val = max(val, start[j, pos - 1] + proc[j, pos - 1])
                pred = pred_on_machine.get((j, pos))
                if pred is not None:
                    pj, pp = pred
                    s = pred_setup.get((j, pos), 0)
                    val = max(val, start[pj, pp] + proc[pj, pp] + s)
                if val != start[j, pos]:
                    start[j, pos] = val
                    changed = True
        if not changed:
            break
    end = start + proc
    return start, end


def schedule_from_machine_seq(routing, proc, machine_seq, family=None, setup=None):
    """Rechnet einen Zeitplan aus einer FESTEN Maschinenreihenfolge (plus Auftragsvorrang) zurück - `None, None`,
    falls die Kombination einen Zyklus schließt (siehe `_is_acyclic_and_longest_path`)."""
    return _is_acyclic_and_longest_path(routing, proc, machine_seq, family, setup)


# --- Kritischer Pfad + Blöcke ------------------------------------------------------------------------------------


def critical_path(routing, proc, start, end, family=None, setup=None):
    """Rückverfolgung vom Operationsende mit dem größten Wert entlang STRAFFER Vorgänger-Kanten (Auftragsvorrang
    ODER Maschinenreihenfolge, je nachdem welche straff ist - mit Rüstzeit heißt "straff" GENAU end[Vorgänger] +
    Rüstzeit) bis zu einer Quelle. Gibt die Operationsfolge Quelle->Senke zurück."""
    n, m = routing.shape
    machine_seq = machine_seq_from_schedule(routing, start)
    pred_on_machine = {}
    pred_setup = {}
    for seq in machine_seq.values():
        for idx in range(1, len(seq)):
            pred_on_machine[seq[idx]] = seq[idx - 1]
            if family is not None:
                pj, pp = seq[idx - 1]
                j2, p2 = seq[idx]
                pred_setup[seq[idx]] = int(setup[family[pj], family[j2]])

    sink = max(((j, pos) for j in range(n) for pos in range(m)), key=lambda jp: end[jp[0], jp[1]])
    path = [sink]
    cur = sink
    while True:
        j, pos = cur
        preds = []
        if pos > 0:
            preds.append(("job", (j, pos - 1), 0))
        mp = pred_on_machine.get((j, pos))
        if mp is not None:
            preds.append(("mach", mp, pred_setup.get((j, pos), 0)))
        tight = [(p, s) for (_, p, s) in preds if abs(end[p[0], p[1]] + s - start[j, pos]) < 1e-6]
        if not tight:
            break
        nxt = None
        for kind, p, s in preds:
            if kind == "mach" and abs(end[p[0], p[1]] + s - start[j, pos]) < 1e-6:
                nxt = p
                break
        if nxt is None:
            nxt = tight[0][0]
        path.append(nxt)
        cur = nxt
    path.reverse()
    return path


def blocks_from_critical_path(routing, path):
    """Maximale Teilfolgen des kritischen Pfads, die auf derselben Maschine liegen."""
    blocks = []
    cur = [path[0]]
    cur_machine = routing[path[0][0], path[0][1]]
    for jp in path[1:]:
        k = routing[jp[0], jp[1]]
        if k == cur_machine:
            cur.append(jp)
        else:
            blocks.append(cur)
            cur = [jp]
            cur_machine = k
    blocks.append(cur)
    return blocks


def n5_neighbors(routing, machine_seq, blocks, exclude_ends=True):
    """Nowicki & Smutnicki (1996): pro Block mit ≥2 Operationen das ERSTE oder LETZTE Paar tauschen - AUSSER dem
    ersten Paar des ERSTEN Blocks und dem letzten Paar des LETZTEN Blocks (literaturübliche Ausnahme, siehe
    Moduldocstring). `exclude_ends=False` lässt die Ausnahme weg: sie ist nur OHNE Rüstzeiten bewiesen (mit
    Rüstzeiten kann ein Tausch am äußersten Paar Rüstzeit sparen und Cmax verkleinern - Orakel-Fund)."""
    moves = []
    for bi, block in enumerate(blocks):
        if len(block) < 2:
            continue
        k = routing[block[0][0], block[0][1]]
        seq = machine_seq[k]
        idx0 = seq.index(block[0])
        if bi != 0 or not exclude_ends:
            new_seq = list(seq)
            i = idx0
            new_seq[i], new_seq[i + 1] = new_seq[i + 1], new_seq[i]
            moves.append((k, new_seq))
        if bi != len(blocks) - 1 or not exclude_ends:
            new_seq = list(seq)
            i = idx0 + len(block) - 2
            new_seq[i], new_seq[i + 1] = new_seq[i + 1], new_seq[i]
            moves.append((k, new_seq))
    return moves


# --- Tabu-Search-Hauptschleife ------------------------------------------------------------------------------------


def tabu_search(routing, proc, start0, end0, max_iters=200, tenure=8, family=None, setup=None, record_trace=False):
    """Startet bei einer gegebenen Lösung (`start0`/`end0`, z. B. Giffler-Thompson/MWKR oder Shifting Bottleneck)
    und verbessert sie über `max_iters` Runden N5-Nachbarschaft + Tabu-Liste. Gibt die BESTE je gefundene Lösung
    zurück (nicht notwendigerweise die letzte, da Tabu Search bewusst auch verschlechternde Züge zulässt, um
    lokale Optima zu verlassen). `record_trace=True` hängt zusätzlich eine Liste jedes neuen Bestwerts an (für die
    Schritt-für-Schritt-Ansicht der App)."""
    n, m = routing.shape
    machine_seq = machine_seq_from_schedule(routing, start0)
    cur_start, cur_end = start0, end0
    cur_cmax = float(cur_end.max())
    best_seq, best_start, best_end, best_cmax = dict(machine_seq), start0, end0, cur_cmax
    tabu = {}
    trace = None
    if record_trace:
        path0 = critical_path(routing, proc, start0, end0, family, setup)
        trace = [{"it": 0, "cmax": cur_cmax, "start": start0, "end": end0, "critical_ops": set(path0)}]

    for it in range(max_iters):
        path = critical_path(routing, proc, cur_start, cur_end, family, setup)
        blocks = blocks_from_critical_path(routing, path)
        moves = n5_neighbors(routing, machine_seq, blocks, exclude_ends=(family is None or not np.any(setup)))
        if not moves:
            break
        cand_best = None       # bester Zug UNTER Beachtung der Tabu-Liste (außer Aspiration)
        fallback_best = None   # bester Zug insgesamt, falls WIRKLICH jeder Zug tabu ist
        for (k, new_seq) in moves:
            trial = dict(machine_seq)
            trial[k] = new_seq
            s2, e2 = schedule_from_machine_seq(routing, proc, trial, family, setup)
            if s2 is None:
                continue
            cmax2 = float(e2.max())
            move_key = (k, tuple(new_seq))
            is_tabu = tabu.get(move_key, -1) > it
            aspiration = cmax2 < best_cmax - 1e-9
            if fallback_best is None or cmax2 < fallback_best[0]:
                fallback_best = (cmax2, k, new_seq, s2, e2, move_key)
            if is_tabu and not aspiration:
                continue
            if cand_best is None or cmax2 < cand_best[0]:
                cand_best = (cmax2, k, new_seq, s2, e2, move_key)
        if cand_best is None:
            # ALLE Kandidaten sind tabu (keine Aspiration) - Standard-Tabu-Search-Praxis: nicht abbrechen,
            # sondern den am wenigsten schlechten tabu-Zug trotzdem nehmen. Ohne diesen Rückfall friert die Suche
            # in genau zwei sich gegenseitig aufhebenden Zügen sofort ein - ein echter Fund beim Bau (README).
            if fallback_best is None:
                break
            cand_best = fallback_best
        cmax2, k, new_seq, s2, e2, move_key = cand_best
        machine_seq[k] = new_seq
        cur_start, cur_end, cur_cmax = s2, e2, cmax2
        tabu[move_key] = it + tenure
        if cur_cmax < best_cmax - 1e-9:
            best_cmax = cur_cmax
            best_seq, best_start, best_end = dict(machine_seq), s2, e2
            if record_trace:
                path2 = critical_path(routing, proc, s2, e2, family, setup)
                trace.append({"it": it + 1, "cmax": best_cmax, "start": s2, "end": e2, "critical_ops": set(path2)})

    machine_finish = np.zeros(m)
    for j in range(n):
        for pos in range(m):
            machine_finish[routing[j, pos]] = max(machine_finish[routing[j, pos]], best_end[j, pos])
    return Result(best_start, best_end, best_cmax, machine_finish, trace)


# --- CP-SAT (exakte Gegenprobe, unverändert aus Stück 8/9) --------------------------------------------------------


def solve_exact(routing, proc, family=None, setup=None, time_limit_seconds=15.0):
    """Ein Kreis-Modell JE MASCHINE - dieselbe exakte Gegenprobe wie in `job-shop-demo`/`shifting-bottleneck-demo`,
    hier eigener Code für dieses Repo."""
    n, m = routing.shape
    if family is None:
        family = np.zeros(n, dtype=np.int64)
        setup = np.zeros((1, 1), dtype=np.int64)
    horizon = int(np.sum(proc)) + int(np.max(setup)) * n * m + 1

    model = cp_model.CpModel()
    start = {}
    end = {}
    for j in range(n):
        for pos in range(m):
            start[j, pos] = model.NewIntVar(0, horizon, f"s{j}_{pos}")
            end[j, pos] = model.NewIntVar(0, horizon, f"e{j}_{pos}")
            model.Add(end[j, pos] == start[j, pos] + int(proc[j, pos]))
            if pos > 0:
                model.Add(start[j, pos] >= end[j, pos - 1])

    ops_per_machine = [[] for _ in range(m)]
    for j in range(n):
        for pos in range(m):
            ops_per_machine[routing[j, pos]].append((j, pos))

    for k in range(m):
        ops = ops_per_machine[k]
        arcs = []
        for idx, (j, pos) in enumerate(ops):
            lit = model.NewBoolVar(f"a0_{k}_{idx}")
            arcs.append((0, idx + 1, lit))
            lit_back = model.NewBoolVar(f"a{k}_{idx}_0")
            arcs.append((idx + 1, 0, lit_back))
        for i2, (j1, p1_) in enumerate(ops):
            for j2idx, (j2, p2_) in enumerate(ops):
                if i2 == j2idx:
                    continue
                lit = model.NewBoolVar(f"a{k}_{i2}_{j2idx}")
                arcs.append((i2 + 1, j2idx + 1, lit))
                s = int(setup[family[j1], family[j2]])
                model.Add(start[j2, p2_] >= end[j1, p1_] + s).OnlyEnforceIf(lit)
        model.AddCircuit(arcs)

    cmax = model.NewIntVar(0, horizon, "cmax")
    model.AddMaxEquality(cmax, list(end.values()))
    model.Minimize(cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = NUM_SEARCH_WORKERS
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, False

    start_arr = np.zeros((n, m))
    end_arr = np.zeros((n, m))
    machine_finish = np.zeros(m)
    for j in range(n):
        for pos in range(m):
            start_arr[j, pos] = solver.Value(start[j, pos])
            end_arr[j, pos] = solver.Value(end[j, pos])
            machine_finish[routing[j, pos]] = max(machine_finish[routing[j, pos]], end_arr[j, pos])
    result = Result(start_arr, end_arr, float(solver.Value(cmax)), machine_finish)
    return result, status == cp_model.OPTIMAL
