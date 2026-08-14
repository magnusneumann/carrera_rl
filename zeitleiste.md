# Zeitleiste: was gemacht wurde und was es verändert hat

Chronologie der Trainingsläufe und der Codeänderungen dazwischen.

**Zweck:** nachvollziehen, welche Läufe überhaupt miteinander vergleichbar sind.
Kurzfassung: erstaunlich wenige.

### Was hier drinsteht

Nur Läufe mit **demselben grundlegenden Verfahren**: ein Agent, der aus drei
gestapelten Graustufenbildern der Vogelperspektive fährt (`n_stack=3`,
`camera_view=global`, vier Graustufen, 166×100). Der **Hauptstrang ist SAC**,
PPO läuft als **Nebenstrang** darunter.

Nicht enthalten sind die frühen Architekturversuche aus dem Juni — Einzelbild,
fahrzeugfeste Ansicht, Hybrid aus Bild und Lidar (`04_` bis `08_`). Die sind
methodisch etwas anderes und nicht auf dieser Achse vergleichbar. Der reine
Lidar-Agent taucht nur als Werkzeug auf, weil er das Backbone liefert.

### Wie die Zahlen zu lesen sind

`best` ist die beste einzelne Auswertung, `Ende` der Mittelwert der letzten
fünf. Beide werden mitgeführt, weil der Unterschied das zentrale Phänomen
dieses Projekts ist (siehe „Das wiederkehrende Muster"). `eplen@best` ist die
Episodenlänge zum Zeitpunkt der besten Auswertung — bei 2000 lief die Episode
ins Zeitlimit.

Durchsatz zur Umrechnung: rund **135 000 Steps/h** für SAC mit Kamera, also
1.5 M ≈ 11 h, 2.5 M ≈ 19 h, 5 M ≈ 33 h.

---

# Hauptstrang: SAC

## Phase 1 — altes Fahrzeugmodell (Juni bis 24.07.)

Physik `force_drag_v1`, alter Reward, `max_steer_change = 1.0`. Das Fahrzeug
beschleunigte mit 12.5 m/s² und bremste mit 14.4 m/s² — rund **20-fach zu
schnell**, was damals niemand wusste.

| Datum | Run | Steps | best | @ | eplen@best | Ende | Abfall |
|---|---|---|---|---|---|---|---|
| 11.06. | `2246` | 720 k | 632.6 | 620 k | 418 | 250.2 | 382 |
| 12.06. | `2347` | 1.5 M | 144.4 | 1.5 M | 151 | 136.3 | 8 |
| 22.06. | `2139` | 1.5 M | 2427.5 | 1.22 M | 1687 | 545.5 | 1882 |
| 24.07. | `1432` | 2.5 M | **3828.2** | 2.0 M | **2000** | −1420.8 | 5249 |

`2347` fällt heraus: dort war `buffer_size` auf 170 k reduziert statt 500 k. Der
Lauf blieb bei 144 und wurde lange als Beleg gelesen, dass ein kleiner Buffer
schadet. Das ist **nicht belegt** — die Gradientenschritte waren mitreduziert,
beide Größen wurden also gleichzeitig verändert.

`1432` ist bis heute das beste Ergebnis des Projekts. Es ist zugleich der
Lauf, der am tiefsten fiel: von 3828 auf −1421.

## Phase 2 — gemessene Physik und neuer Reward (28.–29.07.)

Hier bricht die Vergleichbarkeit zur Phase 1 vollständig ab. **Das Auto ist ein
anderes.**

| Datum | Run | Steps | best | @ | eplen@best | Ende | Abfall |
|---|---|---|---|---|---|---|---|
| 28.07. | `2147` | 2.5 M | −232.0 | 1.93 M | 209 | −359.5 | 127 |
| 29.07. | `2210` | 5 M | 875.7 | 4.07 M | 759 | −315.6 | 1191 |

Die absoluten Zahlen sind gegenüber Phase 1 eingebrochen — was zu erwarten war,
denn das Fahrzeug ist zwanzigmal träger und der Reward zählt anders. `2210`
brauchte **2.78 M Steps bis zur ersten positiven Auswertung** und erreichte den
Höhepunkt erst bei 4.07 M.

`2142` fehlt in der Liste: abgebrochener Start, nur eine Konfigurationsdatei
ohne Modell.

## Phase 3 — gamma-Experiment (31.07.)

| Datum | Run | Steps | best | @ | eplen@best | Ende | Abfall |
|---|---|---|---|---|---|---|---|
| 31.07. | `2331` | 1.5 M | −349.8 | 100 k | 33 | −354.0 | 4 |

Zwei Änderungen zugleich: `w_integral` von 0.08 auf 0.0 und `gamma` von
implizit 0.99 auf 0.999. **Vollständiger Fehlschlag** — der Lauf brach nach
250 k Steps zusammen und lag danach 1.25 M Steps flach bei −355. Der Agent
fuhr 33 Frames weit.

Ursache vermutlich der Wertebereich: mit gamma 0.999 wachsen die zu lernenden
Werte um Faktor 10, die Lernrate blieb bei 5e-5. Hergeleitet in
`rl_erkenntnisse.md`, Abschnitt 4.

## Phase 4 — aktueller Stand (01.08. und 14.08.)

**Noch kein Lauf.** Gegenüber `2331` haben sich drei Dinge geändert:

```
gamma                          0.999  ->  0.99      (zurückgenommen)
Replay-Buffer                  Standard -> Einzelbilder
handle_timeout_termination     False  ->  True
```

Umgebung und Reward sind gegenüber `2331` **unverändert** — Physik, Geometrie,
Kamera und alle zwölf Reward-Parameter stimmen überein, nachgeprüft gegen die
Konfigurationsdatei des Laufs.

---

# Nebenstrang: PPO

Dieselbe Umgebung, dieselbe Beobachtung, anderer Algorithmus. PPO wirft jede
Erfahrung nach einem Update weg und braucht deshalb ein Vielfaches an Schritten
— die Step-Zahlen sind mit dem SAC-Strang **nicht** direkt vergleichbar, die
Laufzeiten schon.

| Datum | Run | lr | BSP | Steps | best | @ | eplen@best | Ende |
|---|---|---|---|---|---|---|---|---|
| 26.07. | `1601` | 3e-4 | ja | 2.4 M | −338.8 | 600 k | 9 | −338.8 |
| 26.07. | `2233` | 3e-4 | nein | 2.5 M | −235.7 | — | — | — |
| 26.07. | `2328` | 1e-4 | nein | 90 M | −3.9 | 54 M | 76 | −79.5 |

Alle drei unter dem **alten** Fahrzeugmodell, also nur untereinander und mit
Phase 1 vergleichbar.

`1601` steht mit einer Episodenlänge von 9 Frames faktisch still. `2328` ist
der einzige PPO-Lauf, der überhaupt etwas gelernt hat — mit 90 M Steps, dem
36-fachen von `1432`, und kam auf −3.9 gegen dessen 3828.

Die Kombination **lr 5e-5 mit Backbone** ist bisher nicht getestet und die
letzte offene Zelle im PPO-Versuchsplan.

---

# Codeänderungen und ihre Wirkung

### 26.07. — Rendering von Pygame auf cv2/numpy

Grund: PPO stürzte mit `n_envs=12` ab, weil jeder Subprozess ein eigenes
Pygame-Fenster öffnete und Windows die Fenster-Handles ausgingen.

**Wirkung auf die Vergleichbarkeit: keine.** Nachgemessen unterscheiden sich
beide Pfade in 0.04 % der Pixel bei identischen Graustufen. Betroffen: alles
bis `1601` lief über Pygame, `2233` und `2328` über cv2.

### 28.07. — Gemessenes Fahrzeugmodell `measured_v2`

Aus Videomessung. Gleichzeitig `max_steer_change` von 1.0 auf 0.5.

**Wirkung: trennt scharf.** Phase 1 und Phase 2 sind nicht vergleichbar.

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

### 31.07. → 01.08. — `gamma` 0.999 und zurück auf 0.99

Siehe Phase 3. `rl_erkenntnisse.md`, Abschnitt 4.

### 14.08. — Replay-Buffer speichert Einzelbilder

`VecFrameStack` legte jedes Bild dreifach ab. Der neue Buffer speichert nur das
neueste und setzt den Stapel beim Ziehen zusammen.

```
Speicher   23.2 GB    ->  7.7 GB      bei 500 000 Übergängen, Faktor 3.00
Zeit       49.1 ms    ->  54.9 ms     je Update, batch 512, GPU   (+11.8 %)
```

**Wirkung auf das Lernen: keine.** Die gezogenen Batches sind bitgleich mit
denen des Standard-Buffers, geprüft über 28 Episodengrenzen und über den Umlauf
des Rings hinweg.

**Wirkung auf die Laufzeit: +11.8 %.** Das war zunächst falsch protokolliert;
die erste Messung lief mit `batch_size=32`. Beim Vergleich über den 14.08.
hinweg bedeutet gleiche Wanduhrzeit ab hier rund 10 % weniger
Gradientenschritte.

### 14.08. — Zeitlimit wird nicht mehr als Endzustand gelernt

`handle_timeout_termination` stand auf `False`, erzwungen durch
`optimize_memory_usage=True`. Damit lernte der Critic, dass bei Frame 2000 die
Welt aufhört und der Zustand wertlos ist — obwohl das Auto dort weiterfuhr.

**Wirkung: trennt scharf.** Erste Änderung seit dem 01.08., die das
Lernverhalten selbst betrifft. Herleitung in `rl_erkenntnisse.md`, Abschnitt
13b. Das Limit von 2000 Frames selbst wurde bewusst nicht angefasst.

---

# Das wiederkehrende Muster

Jeder SAC-Lauf, der überhaupt etwas gelernt hat, erreicht einen Höhepunkt und
fällt danach zurück — meist unter den Ausgangspunkt.

| Run | best | Ende | Abfall | relativ | eplen@best |
|---|---|---|---|---|---|
| `2246` | 632.6 | 250.2 | 382 | 0.60 | 418 |
| `2139` | 2427.5 | 545.5 | 1882 | 0.78 | 1687 |
| `1432` | 3828.2 | −1420.8 | 5249 | **1.37** | **2000** |
| `2210` | 875.7 | −315.6 | 1191 | **1.36** | 759 |
| `2347` | 144.4 | 136.3 | 8 | 0.06 | 151 |
| `2331` | −349.8 | −354.0 | 4 | 0.01 | 33 |

Die beiden Läufe **ohne** Einbruch sind genau die, die nie gut wurden. Das
stützt die Richtung „nur wer etwas kann, kann es verlieren", ist aber trivial:
wer nichts erreicht hat, kann nichts verlieren.

### Was die Zahlen zur Zeitlimit-Hypothese sagen

Der naheliegende Test: fällt tiefer, wessen Episoden ins Zeitlimit liefen?

**Die Daten stützen das nicht eindeutig.** `1432` (eplen 2000) und `2210`
(eplen 759) fielen mit 1.37 und 1.36 praktisch gleich tief. Wäre das Zeitlimit
die alleinige Ursache, müsste `2210` deutlich glimpflicher davonkommen.

Zwei Einschränkungen, die das relativieren:

1. `evaluations.npz` enthält die Episodenlängen der **deterministischen
   Auswertung**, nicht des Trainings. Die Vergiftung entsteht aber aus den
   Trainingsepisoden. Der hier verfügbare Wert ist nur ein Stellvertreter.
2. Der relative Abfall hängt stark am Betrag der Spitze, ist also kein
   sauberes Maß.

**Damit bleibt die Hypothese offen** — plausibler Mechanismus, aber die
vorhandenen Daten belegen ihn nicht. Ein gezielter Versuch ist nötig; siehe
unten.

Weitere Kandidaten: davonlaufender Entropie-Koeffizient, Q-Überschätzung,
Verteilungsdrift im Buffer, Verlust der Plastizität.

---

# Was miteinander vergleichbar ist

| Gruppe | Läufe | gemeinsame Grundlage |
|---|---|---|
| A | `2246`, `2139`, `1432` | altes Modell, alter Reward, SAC |
| B | `1601`, `2233`, `2328` | altes Modell, PPO |
| C | `2147`, `2210` | gemessenes Modell, Reward v3, w_int 0.08 |
| D | `2331` | allein — gamma 0.999 |
| E | *noch leer* | aktueller Stand |

`2347` steht außerhalb: zwei gleichzeitig veränderte Größen.

Zwischen den Gruppen sind nur qualitative Aussagen zulässig („SAC lernt, PPO
kaum"), keine Zahlenvergleiche.

---

# Nächste Schritte

Begründung: eine Änderung pro Lauf, sonst ist die Wirkung nicht zuzuordnen. Und
ohne Referenz unter dem aktuellen Code ist gar nichts zuzuordnen.

| Schritt | Was | Vergleich gegen | Budget |
|---|---|---|---|
| 0a | **SAC auf Lidar, Zeitlimit an/aus** | die zwei Läufe gegeneinander | 2 × 42 min |
| 0b | **SAC mit Kamera, aktueller Stand** | setzt Gruppe E | 30–35 h |
| 1 | Kamera ändern, sonst nichts | Schritt 0b, **gleiche Step-Zahl** | 30–35 h |
| 2 | PPO unter der besseren Kamera | Schritt 0b oder 1, **gleiche Zeit** | 12–24 h |

### Warum Schritt 0a vorgeschaltet ist

Die Zeitlimit-Wirkung zeigt sich erst, wenn Episoden das Limit erreichen — also
erst bei einem guten Agenten. Mit Kamera dauert das **2.78 M Steps bis zur
ersten positiven Auswertung**. Ein kurzer Lauf beantwortet die Frage nicht, ein
vollständiger kostet 30 Stunden je Seite.

SAC auf Lidar-Beobachtung durchläuft dieselbe Mechanik — Replay-Buffer,
Zeitlimit, keine Uhr in der Beobachtung — bei **197 Steps/s**, also 500 k in
42 Minuten. Zwei Läufe mit `handle_timeout_termination` an und aus prüfen den
Mechanismus für rund anderthalb Stunden statt für sechzig.

Das ersetzt den Kameralauf nicht, sondern entscheidet, ob sich die Erwartung
lohnt.

### Warum bei Schritt 1 Steps und bei Schritt 2 Zeit als Maßstab?

Bei der Kamera lautet die Frage „hilft die zusätzliche Information", nicht „ist
sie ihren Rechenaufwand wert". Höhere Auflösung kostet Zeit pro Schritt — bei
Zeitgleichheit würde man sie dafür bestrafen. Gleiche Step-Zahl trennt das.

Bei SAC gegen PPO lautet die Frage dagegen „was bekomme ich für mein
Zeitbudget". PPO braucht zwangsläufig ein Vielfaches an Schritten; gleiche
Step-Zahlen würden es strukturell benachteiligen.

In beiden Fällen werden Steps **und** Laufzeit protokolliert.

### Unabhängig und billig

Ein PPO-Lauf mit `lr 5e-5` und Backbone schließt die letzte leere Zelle im
Versuchsplan (bisher gibt es nur 3e-4 mit Backbone und 1e-4 ohne). Kostet
20–60 Minuten und ist von der obigen Reihenfolge unabhängig.

---

*Stand 14.08.2026. Zahlen aus `models/*/train_config.json` und
`evaluations.npz`.*
