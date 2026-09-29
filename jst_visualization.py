"""Plotly-Abbildungen der Job-Shop-Tabu-Search-Demo: Auftragsübersicht, Mehrzeilen-Gantt (eine Zeile je Maschine,
Farbe nach AUFTRAG, kritischer Pfad schwarz umrandet - die für dieses Stück neue Hervorhebung), Maschinen-Endzeit-
Vergleich, Sweeps, Timing, Start-Vergleich. Achsen sind gesperrt (fixedrange). Das Gantt baut EIN Trace JE
MASCHINE (siehe [[feedback_plotly_many_traces_per_category_shrinks_bars]], von Anfang an richtig gebaut wie in
Stück 7/8/9)."""

import plotly.graph_objects as go

MACHINE_COLORS = ["#4c78a8", "#54a24b", "#e45756", "#f58518", "#b279a2", "#9c755f"]
JOB_COLORS = ["#4c78a8", "#54a24b", "#e45756", "#f58518", "#b279a2", "#9c755f", "#ff9da6", "#9d755d", "#bab0ac", "#eeca3b"]
MWKR_COLOR = "#7f7f7f"
SETUP_COLOR = "#f58518"
SB_COLOR = "#54a24b"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _job_color(j):
    return JOB_COLORS[j % len(JOB_COLORS)]


def build_jobs_chart(routing, proc):
    n, m = routing.shape
    fig = go.Figure()
    for pos in range(m):
        xs, ys, colors, hover = [], [], [], []
        for j in range(n):
            k = int(routing[j, pos])
            xs.append(j)
            ys.append(int(proc[j, pos]))
            colors.append(MACHINE_COLORS[k % len(MACHINE_COLORS)])
            hover.append(f"Auftrag {j}, Schritt {pos + 1}: Maschine {k + 1}, Dauer {proc[j, pos]}")
        fig.add_trace(go.Bar(x=xs, y=ys, marker_color=colors, hovertext=hover, hoverinfo="text", showlegend=False))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Auftrag")
    fig.update_yaxes(title_text="Bearbeitungszeit (gestapelt über alle Operationen)")
    return _base(fig, 280)


def build_machine_gantt(routing, proc, start, end, m, critical_ops=None):
    """EIN Trace JE MASCHINE, Farbe nach AUFTRAG. `critical_ops`: optionale Menge von (j,pos) auf dem kritischen
    Pfad - deren Balken bekommen einen dicken schwarzen Rand statt des dünnen weißen (die für dieses Stück neue
    Hervorhebung, zeigt direkt, worauf Tabu Search seine Züge stützt)."""
    n = routing.shape[0]
    critical_ops = critical_ops or set()
    fig = go.Figure()
    rows = [f"Maschine {k + 1}" for k in range(m)]
    for k in range(m):
        ops_k = [(j, pos) for j in range(n) for pos in range(routing.shape[1]) if routing[j, pos] == k]
        ops_k.sort(key=lambda jp: start[jp[0], jp[1]])
        if not ops_k:
            continue
        xs = [float(proc[j, pos]) for (j, pos) in ops_k]
        bases = [float(start[j, pos]) for (j, pos) in ops_k]
        colors = [_job_color(j) for (j, pos) in ops_k]
        line_colors = ["black" if (j, pos) in critical_ops else "white" for (j, pos) in ops_k]
        line_widths = [3 if (j, pos) in critical_ops else 1 for (j, pos) in ops_k]
        customdata = [j for (j, pos) in ops_k]
        fig.add_trace(go.Bar(x=xs, y=[rows[k]] * len(ops_k), base=bases, orientation="h", width=0.6,
                              marker=dict(color=colors, line=dict(width=line_widths, color=line_colors)),
                              customdata=customdata, showlegend=False, hovertemplate="Auftrag %{customdata}<br>Dauer %{x}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit")
    fig.update_yaxes(categoryorder="array", categoryarray=rows, autorange="reversed")
    return _base(fig, max(160, 40 * m))


def build_machine_finish_comparison(tabu_finish, mwkr_finish):
    m = len(tabu_finish)
    machines = [f"M{k + 1}" for k in range(m)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=machines, y=tabu_finish.tolist(), name="Tabu Search", marker_color=MACHINE_COLORS[0]))
    fig.add_trace(go.Bar(x=machines, y=mwkr_finish.tolist(), name="MWKR-Start (Stück 8)", marker_color=MWKR_COLOR))
    fig.add_hline(y=float(tabu_finish.max()), line=dict(color=MACHINE_COLORS[0], width=1.5, dash="dot"))
    fig.add_hline(y=float(mwkr_finish.max()), line=dict(color=MWKR_COLOR, width=1.5, dash="dot"))
    fig.update_xaxes(title_text="Maschine")
    fig.update_yaxes(title_text="Fertigstellung der letzten Operation")
    fig.update_layout(barmode="group")
    return _base(fig, 320)


def build_sweep(rows, param_label, value_key="value", y_keys=(("gap_mwkr", "MWKR-Start gegenüber Tabu Search", MWKR_COLOR),)):
    xs = [r[value_key] for r in rows]
    fig = go.Figure()
    for key, name, color in y_keys:
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zu Tabu Search (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)


def build_timing(rows):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["exact_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=MWKR_COLOR, width=2.5), name="CP-SAT (volles Modell, im schlimmsten Fall exponentiell)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["tabu_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=MACHINE_COLORS[0], width=2.5), name="Tabu Search (reine Graph-/Kombinatorik-Operationen)"))
    fig.update_xaxes(title_text="Aufträge")
    fig.update_yaxes(title_text="Rechenzeit (ms)", type="log")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 340)


def build_setup_gap(rows):
    xs = [r["value"] for r in rows]
    upper = [r["gap_max"] for r in rows]
    lower = [r["gap_min"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(245,133,24,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap_mean"] for r in rows], mode="lines+markers", line=dict(color=SETUP_COLOR, width=2.5), name="Tabu Search über dem echten Optimum (CP-SAT)"))
    fig.update_xaxes(title_text="Rüstzeit je Familienwechsel (Minuten)")
    fig.update_yaxes(title_text="Abstand zum Optimum (%)")
    return _base(fig, 340)


def build_start_comparison(rows):
    """Das zentrale Experiment dieses Stücks: Tabu Search ab MWKR-Start (Stück 8) gegen ab Shifting-Bottleneck-
    Start (Stück 9), bei fester Budget-Grenze - die beiden Linien liegen fast übereinander (siehe README/Text für
    den überraschenden Befund)."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["from_mwkr"] for r in rows], mode="lines+markers", line=dict(color=MWKR_COLOR, width=2.5), name="Start: MWKR (Stück 8)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["from_sb"] for r in rows], mode="lines+markers", line=dict(color=SB_COLOR, width=2.5, dash="dash"), name="Start: Shifting Bottleneck (Stück 9)"))
    fig.update_xaxes(title_text="Aufträge")
    fig.update_yaxes(title_text="Cmax nach Tabu Search")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 340)
