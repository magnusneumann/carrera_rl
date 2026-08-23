# Zeitleiste: was gemacht wurde und was es verändert hat

Chronologie der Trainingsläufe und der Codeänderungen dazwischen.

**Zweck:** nachvollziehen, welche Läufe überhaupt miteinander vergleichbar sind.
Kurzfassung: erstaunlich wenige.

### Was hier drinsteht

Nur Läufe mit **demselben grundlegenden Verfahren**: ein Agent, der aus drei
gestapelten Graustufenbildern der Vogelperspektive fährt (`n_stack=3`,
`camera_view=global`, vier Graustufen). Der **Hauptstrang ist SAC**, PPO läuft
als **Nebenstrang** darunter.

Die Auflösung ist seit dem 15.08. nicht mehr überall gleich — bis `2331`
166×100, ab `2230` 250×150. Sie steht deshalb bei jedem Lauf dabei.

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

Durchsatz zur Umrechnung, gemessen:

```
166×100   rund 135 000 Steps/h    1.5 M ≈ 11 h   2.5 M ≈ 19 h   5 M ≈ 33 h
250×150   rund  75 000 Steps/h    1.5 M ≈ 20 h
```

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

## Phase 4 — Kontrollversuch (14.08.)

| Datum | Run | Beobachtung | Steps | Dauer | Reward | ep_len | Runden |
|---|---|---|---|---|---|---|---|
| 14.08. | `1748` | **Lidar** | 500 k | 54 min | **6116.4** | 2000 | 11 |

Kein Kameralauf, sondern ein Kontrollversuch: alles konstant gehalten und nur
die Beobachtung gegen exakte Zahlen getauscht. **Der Agent löst die Aufgabe** —
volle 2000 Frames nach 50 k Schritten, über 5000 Reward nach 70 k.

Damit steht fest, woran die Kamera-Agenten scheitern. Alle Modelle unter
`measured_v2` wurden dafür in der heutigen Umgebung neu gefahren
(`python -m src.eval.vergleich`), weil die Zahlen aus `evaluations.npz` aus
verschiedenen Reward-Ständen stammen:

| Modell | Beobachtung | Reward | ep_len | Runden | Tempo |
|---|---|---|---|---|---|
| Lidar-Experte (PPO) | exakt | 6401.7 | 2000 | 11 | 1.22 |
| Lidar SAC (Kontrolle) | exakt | 6116.4 | 2000 | 11 | 1.19 |
| Vision SAC `2210` | 166×100 px | −30.3 | 256 | 0 | 0.81 |
| Vision SAC `2147` | 166×100 px | −266.5 | 174 | 0 | 0.62 |
| Vision SAC `2331` | 166×100 px | −349.8 | 33 | 0 | 0.12 |

Ausgeschlossen sind damit Physik, Reward, Algorithmus, Buffer,
Zeitlimit-Behandlung und Diskontfaktor. Übrig bleibt die **Wahrnehmung**.

Nebenbefund: der Critic-Verlust blieb bei maximal 314 und endete bei 9.6 —
gegen 26 670 bei `1432`. Ein Einbruch zwischen 410 k und 450 k erholte sich von
selbst. SAC ist also nicht grundsätzlich instabil.

Einschränkung: alle Streuungen sind 0, weil Startzustand und Auswertung
deterministisch sind. Die Episoden sind identisch — für belastbare Aussagen
bräuchte es zufällige Startpositionen.

## Phase 5 — höhere Auflösung (15.–16.08.)

| Datum | Run | Auflösung | Steps | Dauer | best | @ | ep_len | Ende |
|---|---|---|---|---|---|---|---|---|
| 15.08. | `2230` | **250×150** | 1.5 M | 20.0 h | **1181.2** | 1.48 M | 769 | 485.8 |

**Der erste Kamera-Agent, der unter `measured_v2` überhaupt Runden fährt.**
Nachgefahren in der heutigen Umgebung (`python -m src.eval.vergleich`):

| Modell | Beobachtung | Reward | ep_len | Runden | Tempo |
|---|---|---|---|---|---|
| Lidar-Experte (PPO) | exakt | 6401.7 | 2000 | 11 | 1.22 |
| **Vision SAC `2230`** | **250×150** | **930.9** | **645** | **2** | **0.97** |
| Vision SAC `2210` | 166×100 | −30.3 | 256 | 0 | 0.81 |
| Vision SAC `2147` | 166×100 | −266.5 | 174 | 0 | 0.62 |

Gegen `2210`, den bisherigen Besten unter derselben Physik:

```
                      2230 (250x150)     2210 (166x100)
beste Auswertung      1181.2 @ 1.48 M     875.7 @ 4.07 M
letzte fuenf           485.8              -315.6
positive Auswertungen  23 von 150 (15 %)   16 von 500 (3 %)
Ertrag je Frame          2.009              1.182
Schwelle (siehe unten)    282 Frames         465 Frames
```

Besser bei einem **Drittel** der Schritte und 20 statt 33 Stunden. Der Ertrag
je Frame hat sich fast verdoppelt — das ist die sauberste Einzelzahl dafür,
dass die Auflösung gewirkt hat, weil sie nicht von der Episodenlänge abhängt.

### Die Sprünge in der Kurve sind kein instabiles Lernen

```
1,400,000    -291.0   ep_len 193     unter der Schwelle
1,410,000   + 807.0   ep_len 656     darueber
1,440,000    -149.5   ep_len 224     wieder darunter
1,480,000   +1181.2   ep_len 769     darueber
```

Der Agent pendelt um die Schwelle von 282 Frames, ab der der Fahrertrag die
feste Crash-Strafe von −300 wieder hereinholt. Die Kennzahl kippt dort, statt
zu steigen. Ausführlich in `rl_erkenntnisse.md`, Abschnitt 10.

### Warum er trotzdem crasht

Reproduzierbar an derselben Stelle — vier von vier beobachteten Episoden,
außen in der linken Haarnadel. Er fährt dort 1.11 m/s, der Experte 1.04, und
bremst 12 Frames vorher statt 40.

Die Ursache ist nicht Sicht und nicht Planungshorizont, sondern die Auflösung
der **Geschwindigkeit**: der Unterschied zwischen 1.04 und 1.11 m/s beträgt in
seinem Bild 0.17 Pixel. Hergeleitet in `rl_erkenntnisse.md`, Abschnitt 13d,
mit `src/eval/fahrlinien.py` als Abbildung.

**Daraus folgt Phase 6.**

## Phase 6 — gespreizter Framestapel (16.–17.08.)

| Datum | Run | Abstand | Steps | best | positiv | letzte fünf |
|---|---|---|---|---|---|---|
| 16.08. | `2307` | **4** | 1.5 M | −134.3 | **0 von 150** | −217.1 |
| 15.08. | `2230` | 1 | 1.5 M | 1181.2 | 23 von 150 | 485.8 |

**Negatives Ergebnis, zurückgenommen.** Der gespreizte Stapel (`t, t−4, t−8`)
lag bei jeder Step-Zahl hinter dem Vorgänger, und der Abstand wuchs:

```
      Step   Abstand 4   Abstand 1    Differenz
   300,000      -361.9      -278.4        -83.5
   700,000      -335.1      -153.6       -181.5
 1,200,000      -281.4       +67.9       -349.3
```

Die Vorhersage war, dass die vervierfachte Verschiebung das Tempo ablesbar
macht. Sie tat es rechnerisch auch — und das Ergebnis wurde trotzdem
schlechter. Zwei Erklärungen, beide ungeprüft:

1. **Zuordnung.** Bei Abstand 1 überlappen sich die drei Autobilder (4.6 px
   groß, 2.5 px Versatz) zu einem Schmierstreifen, dessen Form das Tempo
   codiert. Bei Abstand 4 liegen sie 10 px auseinander — drei getrennte
   Flecken, die das Netz erst einander zuordnen muss.
2. **Positionsunschärfe.** Das älteste Bild ist 0.27 s alt.

**Einschränkung:** der Vergleich hat zwei Variablen. Neben dem Abstand wurde
auch das Backbone neu erzeugt (Datensatz musste zum Abstand passen), dabei
zusätzlich der Experte neu trainiert und der Aufzeichnungspfad von Pygame auf
cv2 umgestellt. Dass es am Abstand lag, ist plausibel, aber nicht bewiesen.

Der Code bleibt (`framestapel_gespreizt.py`, `stride` im Buffer, geprüft für
Abstand 1, 2, 4, 8). `STRIDE = 1` ist die Voreinstellung.

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

## Phase 7 — Startpunkte, Rundenzeit, Auswertung (18.08., vorbereitet)

**Noch kein Lauf.** Drei Änderungen, die nichts an der Aufgabe ändern:

```
START_STREUUNG  = 0.9      Startpunkt streut ueber die Gerade vor der Ziellinie
N_EVAL_EPISODEN = 10       vorher 5
Rundenzeit-Bonus           repariert - hat vorher NIE gefeuert
```

### Der Startpunkt war bisher immer derselbe

Alle Läufe bis einschließlich `2307` starteten bei x = 451 px, 14 Pixel vor der
Ziellinie. In den Konfigurationsdateien ist das inzwischen nachgetragen
(`start_streuung = 0.0`, `env.geometry.start_streuung_m = 0.0`), ebenso
`stride = 1` — beide Felder gab es zur Laufzeit noch nicht.

Die Folge war, dass **jede Episode identisch ablief**: fester Start,
deterministische Physik, bei der Auswertung deterministische Policy. Fünf
Auswertungsepisoden lieferten fünfmal dasselbe, erkennbar an `± 0.00` in allen
Logs. Wir zahlten fünffachen Aufwand für eine einzige Messung und konnten nicht
unterscheiden, ob ein Wert typisch oder ein Glücksfall war.

Jetzt streut der Startpunkt über 0.9 m entlang der Geraden — alle Punkte
weiterhin **vor** der Linie, damit die Rundenzählung ihren Sinn behält.
Richtung und Anfangsgeschwindigkeit bleiben unverändert.

```
vorher   episode_reward = -837.07 +/- 0.00     ep_len 167.00 +/- 0.00
jetzt    episode_reward = -847.68 +/- 9.40     ep_len 196.40 +/- 3.67
         episode_reward = -1166.47 +/- 75.95   ep_len 292.70 +/- 29.53
```

**Wirkung auf die Vergleichbarkeit: vorhanden, aber klein.** Der Agent löst eine
leicht andere Aufgabe. Absolute Zahlen sind mit früheren Läufen nicht mehr
streng vergleichbar.

### Nebenbei zwei Fehler gefunden

**Der Rundenzeit-Bonus hat seit dem 28.07. nie gefeuert.** Die erste
Überquerung nach 9 Frames wurde als Rundenzeit gewertet, `best_lap_frames`
stand danach bei 10, und keine echte Runde (rund 172 Frames) konnte das
unterbieten. `w_lap_improvement = 5.0` war ein toter Parameter. Repariert: die
erste Überquerung bekommt den Rundenbonus, setzt aber keine Zeit.

**Der Seed wurde bei jedem `reset()` neu gesetzt.** Damit hätte der
Zufallsgenerator jedes Mal dieselbe Zahl gezogen und alle Startpunkte wären
trotz Streuung identisch geblieben. Jetzt wirkt der Seed einmal; die Folge der
Ziehungen bleibt reproduzierbar.

---

## Phase 8 — streuende Startpunkte (18.–20.08.)

| Datum | Run | Startpunkt | best | letzte fünf | positiv |
|---|---|---|---|---|---|
| 18.08. | `2254` | fest | 5080.5 | 962.5 | 50 von 150 |
| 20.08. | `2215` | streuend | 1125.2 | 542.7 | 33 von 150 |

Auf dem Papier ein Rückschritt. Tatsächlich das erste ehrliche Messergebnis.

### Der wichtigste Befund dieses Projekts

Beide Modelle unter **beiden** Bedingungen gefahren, je 12 Episoden:

```
                          fester Start        streuender Start
2254 (fest trainiert)          5080.5                 -1829.2
                            9.0 Runden              0.6 Runden

2215 (streuend trainiert)      1027.6                  1545.7
                            2.0 Runden              3.1 Runden
```

`2254` sah mit 5080 großartig aus — **aber nur von seinem einen Startpunkt**.
Aus zufälligen Lagen fällt er auf 0.6 Runden. Er hatte keine Fahrpolitik
gelernt, sondern eine **auswendig gelernte Trajektorie**.

Das ist derselbe Fehlertyp wie beim Vortrainingsnetz (Abschnitt 8 der
Erkenntnisse): perfekt auf der geübten Linie, wertlos daneben. Und es erklärt
rückwirkend den Sprung von 1506 auf 5080 — der Rundenzeit-Bonus belohnt genau
das Perfektionieren einer wiederholten Runde.

`2215` ist unter beiden Bedingungen mittelmäßig, aber konsistent, und der
einzige der beiden, der überhaupt fahren kann statt eine Bahn abzuspulen.

### Was daraus folgt

Alle Kamera-Läufe vor dem 20.08. hatten einen festen Startpunkt. Ihre Zahlen
messen zu einem unbekannten Anteil Auswendiglernen statt Fahren. Das gilt
besonders für die hohen Werte — je besser die Zahl, desto eher war es eine
optimierte Einzeltrajektorie.

**Für alles Weitere: von Anfang an mit streuenden Startpunkten trainieren.**

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
Speicher   23.2 GB    ->  15.5 GB     bei 500 000 Übergängen, Faktor 1.5
Zeit       49.1 ms    ->  54.9 ms     je Update, batch 512, GPU   (+11.8 %)
```

Zur Speicherzahl: der zunächst genannte Faktor 3.00 war gegen einen
Standard-Buffer *ohne* `optimize_memory_usage` gemessen. Die alte Zelle lief
aber **mit** dieser Option, und die legt `next_observations` gar nicht erst an.
Gegen den tatsächlichen Vorzustand bleibt Faktor 1.5. Siehe
`rl_erkenntnisse.md`, Abschnitt 9.

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

### Die Tensorboard-Logs lösen das auf: es sind zwei Fehler

Die obige Tabelle stammt aus `evaluations.npz`. Die Diagnosegrößen im
Tensorboard-Log zeigen ein anderes Bild — ausführlich in
`rl_erkenntnisse.md`, Abschnitt 13d.

**`1432` — der Critic divergiert.** Der Verlust springt unmittelbar nach dem
Höhepunkt von 22.2 auf 26 670, also um Faktor 1200. Kein Verlernen, sondern
ein Stabilitätsereignis.

**`2210` — es gab nie einen Höhepunkt.** Der Trainings-Reward liegt über alle
5 M Schritte zwischen −340 und −226. Die 875.7 in der Tabelle sind eine
einzelne günstige deterministische Auswertung, keine erworbene Fähigkeit.

> **Methodische Konsequenz:** eine Bestauswertung ohne entsprechende Bewegung
> in `rollout/ep_rew_mean` ist ein Ausreißer, kein Können. Die Spalte `best`
> in den Tabellen oben ist entsprechend vorsichtig zu lesen.

### Die Zeitlimit-Hypothese ist damit weitgehend erledigt

Der Vergiftungsmechanismus setzt voraus, dass **Trainingsepisoden** das Limit
erreichen. Bei `1432` auf dem Höhepunkt:

```
Auswertung (deterministisch)   ep_len 2000
Training   (stochastisch)      ep_len  398
```

Im Training crashte der Agent lange vor Frame 2000, weil die Exploration ihn
von der Ideallinie schob. Es landeten also kaum Timeout-Übergänge im Buffer.
Die Korrektur vom 14.08. bleibt richtig, taugt aber **nicht als Erklärung** für
die Einbrüche.

### Warum unter `measured_v2` nichts mehr gelingt

Die Beobachtung ist um Faktor 5 herunterskaliert. Bei 166×100 bewegt sich das
Auto erst ab **0.64 m/s** um mehr als einen Pixel je Frame. Unter dem alten
Modell fuhr es 5–12 m/s, also 8–19 px je Frame; unter `measured_v2` sind
höchstens 3 px möglich, und unterhalb von 0.64 m/s sind die drei gestapelten
Bilder identisch.

Der Agent kann seine Geschwindigkeit in genau dem Bereich nicht wahrnehmen, in
dem er startet — und wird vom Reward dafür bestraft (`v_slow = 0.6`).

Der Lidar-Experte löst dieselbe Aufgabe unter derselben Physik und demselben
Reward in 12 Minuten. Das grenzt die Ursache ein: nicht Physik, nicht Reward,
nicht der Algorithmus, sondern die **Wahrnehmung**.

`measured_v2` hat das nicht verursacht, sondern aufgedeckt. Das alte Modell
verdeckte es durch ein zwanzigfach zu schnelles Fahrzeug.

---

# Was miteinander vergleichbar ist

| Gruppe | Läufe | gemeinsame Grundlage |
|---|---|---|
| A | `2246`, `2139`, `1432` | altes Modell, alter Reward, SAC |
| B | `1601`, `2233`, `2328` | altes Modell, PPO |
| C | `2147`, `2210` | gemessenes Modell, Reward v3, w_int 0.08 |
| D | `2331` | allein — gamma 0.999 |
| E | `2230` | 250×150, Einzelbild-Buffer, Zeitlimit korrekt |
| F | *noch leer* | zusätzlich gespreizter Framestapel |

`2347` steht außerhalb: zwei gleichzeitig veränderte Größen.

Zwischen den Gruppen sind nur qualitative Aussagen zulässig („SAC lernt, PPO
kaum"), keine Zahlenvergleiche.

---

# Nächste Schritte

Begründung: eine Änderung pro Lauf, sonst ist die Wirkung nicht zuzuordnen. Und
ohne Referenz unter dem aktuellen Code ist gar nichts zuzuordnen.

Schritt 0 ist **erledigt**: die Ursache ist die Wahrnehmung (Phase 4).
Die Auflösung läuft gerade (Phase 5).

| Schritt | Was | Beantwortet | Budget |
|---|---|---|---|
| ~~0~~ | ~~SAC auf Lidar~~ | ~~liegt es an der Wahrnehmung?~~ | **erledigt** |
| ~~1~~ | ~~Auflösung 250×150~~ | ~~hilft mehr Bildschärfe?~~ | **läuft** |
| 2 | **Framestapel zeitlich spreizen** | hebt es die Subpixel-Grenze auf? | 12–24 h |
| 3 | `crash_penalty` senken | löst es die Schwelle auf? | 12–24 h |
| 4 | SAC **ohne** Backbone | ist das Vortraining seinen Aufwand wert? | 12–24 h |
| 5 | PPO unter der besseren Kamera | SAC gegen PPO, **gleiche Zeit** | 12–24 h |

### Warum Schritt 2 vor Schritt 3 kommt

Schritt 2 greift das einzige **bewiesene** Problem an (Wahrnehmung, belegt
durch den Lidar-Kontrollversuch und die Ablation) und kostet weder Speicher
noch Rechenzeit. Schritt 3 greift ein diagnostiziertes, aber nicht als Ursache
bewiesenes Problem an und ändert dabei die Reward-Skala — danach sind die
Zahlen wieder nicht mit früheren Läufen vergleichbar.

### Warum Schritt 4 dazugekommen ist

Das vortrainierte Netz konnte in keiner Fassung fahren (24, 68 und 230 Frames,
nie eine Runde — siehe `rl_erkenntnisse.md`, Abschnitt 8). Lauf `1432` gelang
trotzdem, die Faltungsschichten taugen also als Augen. Ob sie mehr nützen als
zufällige Anfangsgewichte, ist für SAC **nie geprüft** worden. Für PPO gibt es
den `bsp`-Vergleich, für SAC nicht.

Solange das aussteht, kostet jeder Lauf einen Datensatz und ein Vortraining,
ohne dass belegt wäre, dass es etwas bringt.

### Warum Schritt 1 jetzt vorne steht

Die Subpixel-Rechnung nennt die Grenze: unter **0.64 m/s** bewegt sich das Auto
weniger als einen Pixel je Frame, die drei gestapelten Bilder sind dann
identisch. Zwei Hebel:

```
Framestapel spreizen    t, t-4, t-8  statt  t, t-1, t-2
                        bei 0.3 m/s: 3.8 px Versatz statt 0.47
                        Schwelle faellt von 0.64 auf 0.16 m/s

Aufloesung erhoehen     300x180 statt 166x100
                        Schwelle faellt von 0.64 auf nur 0.35 m/s
```

Das Spreizen ist der stärkere und der billigere Hebel — es kostet weder
Speicher noch Rechenzeit, während höhere Auflösung beides kostet. Deshalb
zuerst.

Umsetzbar über den Replay-Buffer, der die Stapel ohnehin aus Indexversätzen
zusammensetzt (`framestapel_buffer.py`): aus `batch_inds - z` wird
`batch_inds - z*schritt`. Die Umgebung muss dafür ebenfalls gespreizt liefern,
sonst weichen Training und Anwendung voneinander ab.

### Was der Kontrollversuch offengelassen hat

Die **Zeitlimit-Korrektur** ist weiterhin nicht isoliert geprüft. Sie war in
diesem Lauf aktiv, und der Lauf hielt 450 k Schritte ohne Zusammenbruch — das
ist aber kein Beleg, weil die Gegenprobe fehlt.

Die **Critic-Divergenz** aus `1432` trat hier nicht auf. Ob sie unter Kamera
wiederkommt, zeigt erst Schritt 1.

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
