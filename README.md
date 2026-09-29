# Job-Shop-Tabu-Search – Streamlit-Demo

**[→ Demo live ausprobieren](#) (Deploy offen)**

Zehntes Stück der **Klassische-Scheduling-Theorie-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch
– Operations Research und Machine Learning", das zweite SOTA-Stück: dasselbe Job-Shop-Modell wie Stück 8/9, aber
statt einmal gut zu KONSTRUIEREN (Shifting Bottleneck, Stück 9) verbessert **Tabu Search** (Nowicki & Smutnicki
1996) eine gegebene Startlösung (MWKR, Stück 8) iterativ über eine gezielte, kleine Nachbarschaft.

**Einordnung in die Linie:**
```
                                                             ┌─ Shifting Bottleneck (Stück 9, konstruiert einmal gut)
Job Shop (Giffler-Thompson + MWKR, Stück 8 - Konvergenzpunkt) ┤
                                                             └─ Job-Shop-Tabu-Search (dieses Stück, verbessert iterativ)
```
Auf dem **kritischen Pfad** (dem Weg ohne jeden Puffer) wird in **Blöcke** zerlegt (maximale Teilfolgen auf
derselben Maschine); pro Block wird nur das ERSTE oder LETZTE Operationspaar vertauscht (**N5-Nachbarschaft**,
Nowicki & Smutnicki 1996, WebSearch-verifiziert) - mit einer Ausnahme aus der Originalarbeit: das erste Paar des
allerersten Blocks und das letzte Paar des allerletzten Blocks werden NICHT vertauscht (diese Züge können Cmax
beweisbar nicht verbessern). Eine winzige, gezielte Nachbarschaft im Vergleich zur vollen paarweisen Nachbarschaft
der Schwester-Demo `tabu-search-demo` (Trajektorien-Metaheuristiken-Linie, TSP).

Ergebnis in Kürze: bei 10 Aufträgen auf 4 Maschinen (Standard-Seed) verbessert Tabu Search den MWKR-Start um
**18,7 %**. Über die feste Messreihe (5 Instanzen) hilft die Suche im Mittel **7,9 %**. **Der überraschendste
Befund dieses Stücks**: zählt ein besserer Start (Shifting Bottleneck, Stück 9) gegenüber dem naiven MWKR-Start
(Stück 8) bei fester Budget-Grenze? Gemessen NICHT bestätigt - bei den meisten Instanzen konvergieren beide Starts
zum fast identischen Ergebnis (Gleichstand-Quote bis zu 100 %), teils sogar mit dem SCHLECHTEREN Start am Ende
knapp vorn. Das widerspricht der ursprünglichen Erwartung (aus der Schwester-Tabu-Demo für TSP, wo ein guter Start
stark zählt) - die N5-Nachbarschaft scheint robust genug, einen schwächeren Start selbst auszugleichen.

| Frage | Ergebnis (Mittel über 5 feste Instanzen, Seeds 100000–100004) |
|---|---|
| Standardfall (10 Aufträge, 4 Maschinen, Seed 60) | ✅ Tabu Search **18,7 %** unter dem MWKR-Start |
| **Verbesserung im Mittel** | ✅ **+7,9 %** gegenüber dem MWKR-Start |
| **Zählt ein besserer Start (Shifting Bottleneck, Stück 9)?** | ❌ Kaum - Gleichstand-Quote bis zu 100 % |
| **Tabu Search auf JEDER Instanz mindestens so gut wie der Start?** | ✅ Ja - bewiesen, nicht nur gemessen (siehe unten) |
| **N5-Nachbarschaft bietet immer einen Zug?** | ❌ Nein - liegt der GESAMTE kritische Pfad auf einer Maschine, kein Zug möglich |

## Was die Demo zeigt

1. **Tabu Search in Aktion** (Schritt-Slider): **Aufträge** (gestapelter Balken je Auftrag in eigener
   Maschinenreihenfolge) → **Kritischer Pfad & Blöcke** (Regler "Verbesserung" - 0 = MWKR-Start, jeder weitere
   Schritt eine gefundene Verbesserung; Gantt mit einer Zeile JE MASCHINE, Farbe nach AUFTRAG, kritischer Pfad
   SCHWARZ UMRANDET) → **Ergebnis** (Fertigstellung je Maschine, Tabu Search gegen den MWKR-Start).
2. **Was die Verbesserung bringt:** Tabu Search (Cmax), Abstand des MWKR-Starts, CP-SAT-Gegenprobe (n ≤ 8, mit
   Beweis-Status).
3. **📐 Sweep** über die Anzahl der Aufträge ODER Maschinen.
4. **🔬 Experimente auf Abruf:** zählt ein besserer Start (der zentrale, überraschende Befund dieses Stücks);
   Rechenzeit CP-SAT (volles Modell) gegen Tabu Search (reine Graph-/Kombinatorik-Operationen, kein CP-SAT im
   Hauptfluss); Rüstzeit-Härtetest.
5. **🚧 Grenzen:** Tabelle mit Verweis auf das letzte Stück der Linie (RCPSP).

Regler: Aufträge (2–20), **Maschinen** (2–6), **Vehikel** (Neutral/Werkstatt-Logistik – bei Werkstatt zusätzlich
Rüstzeit und Anzahl Familien), Seed der Instanz (+ 🎲), **Budget (Iterationen)**, **Tabu-Tenure**.

## Die zwei Vehikel (gelten für die ganze Linie)

- **Neutral** (`jst_scenario.py`): dasselbe Instanzmodell wie Stück 8/9 - $n$ Aufträge, jeder mit einer
  zufälligen Permutation der $m$ Maschinen als eigener Reihenfolge, Bearbeitungszeiten $\sim U(1, 100)$ je
  Operation.
- **Werkstatt/Logistik** (`jst_scenario_logistik.py`): dieselbe Instanz, aber jeder Auftrag gehört zu einer
  Familie; ein Familienwechsel kostet eine feste Rüstzeit JE MASCHINE - der kritische Pfad berücksichtigt sie
  mit ("straff" heißt dann `start(Nachfolger) = end(Vorgänger) + Rüstzeit`). Rüstzeit 0 kollabiert strukturell
  exakt zum neutralen Vehikel (per Test belegt).

## Modell und Verfahren

- **Instanz** (`jst_scenario.py`, `jst_scenario_logistik.py`): Routing, Bearbeitungszeiten, Familien und
  Rüstzeit-Matrix - identisch zu Stück 8/9, eigener Code für dieses Repo.
- **Kritischer Pfad + Blöcke** (`jst_algorithm.critical_path`/`blocks_from_critical_path`): Rückverfolgung
  entlang straffer Vorgänger-Kanten (Auftragsvorrang ODER Maschinenreihenfolge), dann Zerlegung in maximale
  Teilfolgen auf derselben Maschine.
- **N5-Nachbarschaft** (`jst_algorithm.n5_neighbors`): pro Block das erste/letzte Paar - mit der literaturüblichen
  Ausnahme an den beiden äußersten Blöcken.
- **Tabu-Search-Hauptschleife** (`jst_algorithm.tabu_search`): bester erlaubter Zug je Runde (Aspiration erlaubt
  Ausnahmen), Rückfall auf den am wenigsten schlechten TABU-Zug, falls wirklich alle gesperrt sind (echter Fund,
  siehe unten). Gibt die BESTE je gefundene Lösung zurück.
- **Zweite Startlösung** (`jst_shifting_bottleneck.py`): Shifting Bottleneck (Adams/Balas/Zawack 1988), kopiert
  aus `shifting-bottleneck-demo` (Stück 9) für das Start-Vergleichs-Experiment - jedes Stück dieser Linie ist
  eigenständig lauffähig, deshalb Kopie statt Cross-Repo-Import.
- **CP-SAT** (`jst_algorithm.solve_exact`): dasselbe Kreis-Modell wie Stück 8/9, hier als exakte Gegenprobe.
  Gegen unabhängige Brute-Force-Vollaufzählung verifiziert.
- **Auswertung** (`jst_evaluation.py`): Kennzahlen, Sweep, Start-Vergleich, Timing-Messreihe, Rüstzeit-Härtetest.

## Was nicht funktioniert hat / Grenzen

- **Ein echter Bug beim Bau: alle Züge tabu ließ die Suche sofort einfrieren.** Bei n=6, m=3, Seed 3 hatte der
  kritische Pfad nur EINEN möglichen N5-Zug; die erste Runde machte ihn, die zweite Runde hätte ihn exakt
  rückgängig gemacht - aber dieser Rückweg war (korrekt) tabu, und es gab keinen anderen Zug. Die ursprüngliche
  Implementierung brach die gesamte Suche in diesem Fall sofort ab (0 % Verbesserung über 1000 Iterationen, JEDE
  getestete Tenure). **Fix**: Standard-Tabu-Search-Praxis - sind wirklich ALLE Kandidaten tabu (keine Aspiration),
  wird trotzdem der am wenigsten schlechte davon gemacht, statt die Suche zu beenden. Mit dem Fix erreicht genau
  diese Instanz sogar das CP-SAT-Optimum. Regressionstest: `test_all_tabu_moves_fall_back_instead_of_freezing_the_search`.
- **Vorab-Annahme: "ein besserer Start (Shifting Bottleneck) hilft bei festem Budget deutlich"** - **gemessen
  NICHT bestätigt** (siehe Ergebnis-Tabelle oben). Bei n=15/n=20 sind MWKR-Start und Shifting-Bottleneck-Start
  über alle 5 Sweep-Seeds hinweg exakt gleichauf; bei n=10 gewinnt sogar der SCHWÄCHERE MWKR-Start knapp. Anders
  als bei der Schwester-Tabu-Demo für TSP (dort zählt der Start stark, siehe deren README) scheint die
  N5-Nachbarschaft mächtig genug, einen schwächeren Start bei diesem Budget selbst auszugleichen.
- **Die N5-Nachbarschaft bietet nicht immer einen Zug.** Liegt der GESAMTE kritische Pfad auf EINER Maschine (ein
  einziger Block), greift die Ausnahmeregel an BEIDEN Enden gleichzeitig (erster UND letzter Block zugleich) -
  kein einziger Zug ist erlaubt. Kein Fehler, ein dokumentierter Grenzfall der Nachbarschaft selbst
  (`test_single_critical_block_offers_zero_moves`).
- **Tabu Search verschlechtert die Startlösung nie** - anders als Stück 7-9 dieser Linie (LPT/MWKR/Shifting
  Bottleneck können auf einzelnen Instanzen schlechter abschneiden als ihre Vergleichsregel) ist das hier eine
  BEWIESENE Garantie: die Funktion gibt immer die beste je gefundene Lösung zurück, im schlimmsten Fall also die
  Startlösung selbst (`test_tabu_search_never_makes_the_start_worse`).
- **Synthetische Instanzen:** wie Stück 8/9 - Bearbeitungszeiten gleichverteilt, jeder Auftrag besucht jede
  Maschine genau einmal.

## Verifikation

- **CP-SAT gegen unabhängige Brute-Force-Vollaufzählung** (mit und ohne Rüstzeiten): für n ≤ 4, m ≤ 3 über
  mehrere Seeds stimmt das Kreis-Modell exakt mit einer unabhängigen Vollaufzählung überein.
- **Tabu Search gegen CP-SAT-Optimum**: für n = 2 bis 4 (mehrere Seeds) liegt Tabu Search nie unter dem
  bewiesenen Optimum.
- **Kritischer Pfad und Blöcke per Handrechnung**: eine 2-Auftrags/2-Maschinen-Instanz und eine von Hand
  konstruierte Blockzerlegung bestätigen die Konstruktion; die N5-Ausnahmeregel ist direkt getestet (erstes
  Paar des ersten Blocks, letztes Paar des letzten Blocks, mittlerer Block, Ein-Operationen-Block).
- **Struktureller Konsistenz-Test**: Tabu Searchs Zeitplan bei Rüstzeit 0 ist identisch mit dem neutralen
  Vehikel.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Standardfall, Start-Vergleich, Rüstzeit-Härtetest);
  alle 5 Presets geprüft; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt auf beiden Vehikeln,
  Würfel-Knopf, Permalink-Grenzen inkl. ungültigem Vehikel, Extremwerte inkl. Budget/Tenure, Experimente auf
  Abruf, Footer); eigener Test für die ausblendbaren Regler.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweep, 🔬 Experimente, 🚧 Grenzen, Mathe |
| `jst_algorithm.py` | Kritischer Pfad, Blöcke, N5-Nachbarschaft, Tabu-Search-Hauptschleife, Giffler-Thompson/MWKR (Startlösung), CP-SAT |
| `jst_shifting_bottleneck.py` | Shifting Bottleneck (Stück 9, kopiert) als zweite Startlösung für das Start-Vergleichs-Experiment |
| `jst_scenario.py` | Vehikel Neutral |
| `jst_scenario_logistik.py` | Vehikel Werkstatt/Logistik (Familien, Rüstzeit-Matrix) |
| `jst_constants.py` | Konstanten, Presets |
| `jst_evaluation.py` | Kennzahlen, Sweep, Start-Vergleich, Timing-Messreihe, Rüstzeit-Härtetest |
| `jst_presets.py`, `jst_visualization.py` | Permalink/Presets (inkl. `seed_widget`/`KEPT` für ausblendbare Regler), Plotly-Figuren (ein Trace je Maschine, kritischer Pfad schwarz umrandet, achsengesperrt) |
| `tests/` | CP-SAT gegen Vollaufzählung, Tabu Search gegen CP-SAT, kritischer Pfad/Blöcke/N5 per Handrechnung, Szenario und Auswertung, Aussagen der App, Presets, versteckter Widget-Zustand, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
