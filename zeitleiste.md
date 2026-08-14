# Zeitleiste: Änderungen und ihre Wirkung

Chronologie aller Trainingsläufe mit dem jeweils zugehörigen Stand von Physik,
Reward und Hyperparametern. Ergänzt um die Codeänderungen dazwischen, die den
Vergleich zwischen den Läufen beeinflussen.

**Zweck:** nachvollziehen, welche Läufe überhaupt miteinander vergleichbar sind.
Kurzfassung: erstaunlich wenige.

`best` ist der höchste Wert einer einzelnen Auswertung, `Ende` der Mittelwert
der letzten fünf. Der Unterschied ist erheblich und beabsichtigt mitgeführt —
siehe Abschnitt „Wiederkehrendes Muster" unten.

---

## Läufe

| Datum | Run | Algo | Physik | Reward | w_int | gamma | BSP | Steps | best | Ende |
|---|---|---|---|---|---|---|---|---|---|---|
| 06.11 | 2246 | SAC | force_drag_v1 | alt | 0.0 | implizit | ja | 720 k | 632.6 | 250.2 |
| 06.12 | 2347 | SAC | force_drag_v1 | alt | 0.0 | implizit | ja | 1.5 M | 144.4 | 136.3 |
| 06.22 | 2139 | SAC | force_drag_v1 | alt | 0.0 | implizit | ja | 1.5 M | 2427.5 | 545.5 |
| 07.24 | 1432 | SAC | force_drag_v1 | alt | 0.0 | implizit | ja | 2.5 M | **3828.2** | −1420.8 |
| 07.26 | 1601 | PPO | force_drag_v1 | alt | 0.0 | implizit | ja | 2.4 M | −338.8 | −338.8 |
| 07.26 | 2233 | PPO | force_drag_v1 | alt | 0.0 | implizit | **nein** | 2.5 M | −235.7 | — |
| 07.26 | 2328 | PPO | force_drag_v1 | alt | 0.0 | implizit | **nein** | 90 M | −3.9 | −79.5 |
| 07.28 | 2147 | SAC | **measured_v2** | **v3** | 0.08 | implizit | ja | 2.5 M | −232.0 | −359.5 |
| 07.29 | 2210 | SAC | measured_v2 | v3 | 0.08 | implizit | ja | 5 M | 875.7 | −315.6 |
| 07.31 | 2331 | SAC | measured_v2 | v3 | **0.0** | **0.999** | ja | 1.5 M | −349.8 | −354.0 |

`2142` fehlt: abgebrochener Start, nur eine Konfiguration ohne Modell.

---

## Codeänderungen dazwischen

### 26.07., 22:28 — Rendering von Pygame auf cv2/numpy

Grund: PPO stürzte mit `n_envs=12` ab, weil jeder Subprozess ein eigenes
Pygame-Fenster öffnete und Windows die Fenster-Handles ausgingen.

**Wirkung auf die Vergleichbarkeit: keine.** Nachgemessen unterscheiden sich
beide Pfade in 0.04 % der Pixel bei identischen Graustufen. Läufe davor und
danach sind vergleichbar.

Betroffen: alles bis `1601` lief über Pygame, `2233` und `2328` über cv2.

### 28.07. — Gemessenes Fahrzeugmodell `measured_v2`

Aus Videomessung. Das alte Modell beschleunigte mit 12.5 m/s² und bremste mit
14.4 m/s², gemessen sind 0.25 und 0.68 — alle Zeitkonstanten rund 20-fach zu
kurz. Gleichzeitig `max_steer_change` von 1.0 auf 0.5.

**Wirkung: trennt scharf.** Läufe bis `1601`/`2328` sind mit `2147` und später
nicht vergleichbar. Das Auto ist ein anderes.

### 28.07. — Reward `v3_rundenzeit`

Drei Änderungen zugleich: Gefälle statt Klippe im langsamen Bereich, Strafe für
langsamere Runden entfernt (sie machte eine einzelne schnelle Runde netto −250
wert), tote Parameter entsorgt.

**Wirkung: trennt scharf.** Reward-Zahlen vor und nach dem 28.07. bedeuten
Verschiedenes.

### 31.07. — `w_integral` 0.08 → 0.0

Der Oszillationsschutz besteuerte jedes Lenken statt nur Zappeln: der gut
fahrende Lidar-Experte zahlte damit −625 über 2000 Frames, mehr als seine
gesamten Rundenboni.

**Wirkung: trennt `2147`/`2210` von `2331`.**

### 31.07. — `gamma` explizit, 0.99 → 0.999

Begründung war der Bremsweg: das Auto braucht aus 1.3 m/s rund 97 Frames, der
Horizont bei 0.99 beträgt 100 Frames.

**Das war ein Fehlschlag.** Lauf `2331` brach nach 250.000 Steps zusammen und
lag danach 1.25 Millionen Steps flach bei −355. Vermutliche Ursache: die zu
lernenden Werte wachsen um Faktor 10 (von ~290 auf ~2900), die Lernrate blieb
bei 5e-5.

### 01.08. — `gamma` zurück auf 0.99

Begründung siehe `rl_erkenntnisse.md`, Abschnitt 4.

### 14.08. — Replay-Buffer speichert Einzelbilder (`4f964ce`, `7a4a616`)

`VecFrameStack` legte jedes Bild dreifach ab. Der neue Buffer speichert nur das
neueste und setzt den Stapel beim Ziehen zusammen.

```
Speicher   23.2 GB    ->  7.7 GB      bei 500 000 Übergängen, Faktor 3.00
Zeit       49.1 ms    ->  54.9 ms     je Update, batch 512, GPU   (+11.8 %)
```

**Wirkung auf das Lernen: keine.** Die gezogenen Batches sind bitgleich mit
denen des Standard-Buffers, geprüft über 28 Episodengrenzen und über den
Umlauf des Rings hinweg.

**Wirkung auf die Laufzeit: +11.8 %.** Das war ursprünglich falsch
protokolliert. Die erste Messung lief mit `batch_size=32` und ergab 12.9 gegen
12.8 s; bei der echten `batch_size=512` tritt der Aufschlag hervor. Er steckt
vollständig im Ziehen (8.6 → 13.9 ms je Batch), weil der Stapel dort erst
zusammengesetzt wird. Eine erste Fassung lag bei 26 ms; der Aufbau wurde auf
einen einzigen Zugriff mit einem Indexfeld der Form (N, n_stack) umgestellt.

Beim Vergleich von Laufzeiten über den 14.08. hinweg ist das zu
berücksichtigen: gleiche Wanduhrzeit bedeutet ab hier rund 10 % weniger
Gradientenschritte.

Nachgereicht am selben Tag: an der Naht des Ringpuffers griff die
Rekonstruktion auf überschriebene Vorgänger zu (2 von 500 000 Indizes). Beide
ursprünglichen Tests liefen mit halbleerem Buffer und konnten das nicht sehen.
Siehe `rl_erkenntnisse.md`, Abschnitt 13c.

### 14.08. — Zeitlimit wird nicht mehr als Endzustand gelernt (`4f964ce`)

`handle_timeout_termination` stand auf `False`, erzwungen durch
`optimize_memory_usage=True`. Damit lernte der Critic, dass bei Frame 2000 die
Welt aufhört und der Zustand wertlos ist — obwohl das Auto dort weiterfuhr.

**Wirkung: trennt scharf.** Das ist die erste Änderung seit dem 01.08., die das
Lernverhalten selbst betrifft. Betroffen sind ausschließlich Läufe, deren
Episoden das Limit erreichten — also gerade die guten. `1432` hatte auf dem
Höhepunkt eine Episodenlänge von exakt 2000.

Ob das die wiederkehrenden Einbrüche nach dem Höhepunkt erklärt, ist **offen**.
Der Mechanismus ist in `rl_erkenntnisse.md`, Abschnitt 13b hergeleitet; der
Referenzlauf (Schritt 0 unten) prüft ihn.

Das Limit von 2000 Frames selbst wurde bewusst **nicht** angefasst. Es stammt
aus dem ersten PPO-Commit vom 18.05. und damit aus der Zeit vor der
Physikkorrektur, ist unter `measured_v2` aber weiterhin großzügig: eine Runde
misst 7.66 m, bei Vollgas sind 15.8 Runden im Limit möglich, für eine einzelne
Runde genügen 0.115 m/s im Schnitt.

**Aktueller Codestand. Entspricht keinem der bisherigen Läufe.**

---

## Was daraus folgt

### Vergleichbare Gruppen

```
A  2246, 2347, 2139, 1432, 1601        force_drag_v1 + alter Reward
B  2233, 2328                          dito, aber ohne BSP
C  2147, 2210                          measured_v2 + v3 + w_integral 0.08
D  2331                                measured_v2 + v3 + w_integral 0.0 + gamma 0.999
E  (noch keiner)                       aktueller Stand
```

Innerhalb von A ist `1432` gegen `1601` das einzige sauber gepaarte
SAC-gegen-PPO-Duell: gleiche Physik, gleicher Reward, beide mit Backbone,
2.5 M gegen 2.4 M Steps. Ergebnis 3828 gegen −339.

**Gruppe E ist leer.** Solange dort nichts liegt, ist keine weitere Änderung
messbar — es fehlt der Bezugspunkt.

### Wiederkehrendes Muster: Spitze, dann Absturz

Jeder SAC-Lauf erreicht einen Höhepunkt und fällt danach ab:

```
1432:  3828 → −1421      (Abfall 5249)
2139:  2428 →   546
2210:   876 →   −316
2246:   633 →    250
```

Das gilt ausnahmslos und über alle Konfigurationen hinweg. Für die Auswertung
heißt das: **`best` und `Ende` gehören beide in jede Tabelle.** Nur den
Spitzenwert zu berichten überzeichnet, nur den Endwert unterschlägt, was
erreichbar war.

Verschärfend: die fünf Eval-Episoden sind bei sieben von neun Läufen zu 100 %
identisch (deterministische Umgebung, `deterministic=True`). Jeder Punkt ist
also **eine** Episode, kein Mittelwert. Der Spitzenwert 3828 ist ein einzelner
Durchlauf.

Ausnahmen: `2147` (2 % identisch) und `2210` (1 %), beide unter `measured_v2`.
Ursache ungeklärt — Vermutung ist die Nullstellen-Klemme im neuen Modell, an
der Gleitkomma-Unterschiede aufgeblasen werden.

---

## Geplante Reihenfolge

Begründung: eine Änderung pro Lauf, sonst ist die Wirkung nicht zuzuordnen.
Und ohne Referenz unter dem aktuellen Code ist gar nichts zuzuordnen.

| Schritt | Was | Vergleich gegen | Budget |
|---|---|---|---|
| 0 | **SAC, aktueller Stand** | — (setzt die Referenz) | 12–24 h |
| 1 | Kamera ändern, sonst nichts | Schritt 0, **gleiche Step-Zahl** | 12–24 h |
| 2 | PPO unter der besseren Kamera | Schritt 0 oder 1, **gleiche Zeit** | 12–24 h |

**Warum bei Schritt 1 Steps und bei Schritt 2 Zeit als Maßstab?**

Bei der Kamera lautet die Frage „hilft die zusätzliche Information", nicht „ist
sie ihren Rechenaufwand wert". Höhere Auflösung kostet Zeit pro Schritt — bei
Zeitgleichheit würde man sie dafür bestrafen. Gleiche Step-Zahl trennt das.

Bei SAC gegen PPO lautet die Frage dagegen „was bekomme ich für mein
Zeitbudget". PPO wirft jede Erfahrung nach einem Update weg und braucht
zwangsläufig ein Vielfaches an Schritten; gleiche Step-Zahlen würden es
strukturell benachteiligen.

In beiden Fällen werden Steps **und** Laufzeit protokolliert, dann lässt sich
hinterher beides berichten.

### Nebenbei, unabhängig und billig

Ein PPO-Lauf mit `lr 5e-5` und Backbone schließt die letzte leere Zelle im
Versuchsplan (bisher gibt es nur 3e-4 mit Backbone und 1e-4 ohne). Kostet
20–60 Minuten und ist von der obigen Reihenfolge unabhängig.

### Erwartete Dauer von Schritt 0

Unter `measured_v2` erreichte SAC seine erste positive Auswertung erst bei
2.78 M Steps (~18.5 h) und den Höhepunkt bei 4.07 M (~27 h). Ein 6-Stunden-Lauf
(900 k Steps) bliebe deutlich darunter.

Für einen reinen A/B-Vergleich ist das verschmerzbar — beide Seiten müssen nur
dasselbe Budget bekommen, nicht unbedingt auskonvergieren. Für eine Aussage
über das Erreichbare braucht es die 24 Stunden.
