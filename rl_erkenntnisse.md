# RL-Erkenntnisse aus diesem Projekt

Gesammelte theoretische Punkte, jeweils mit den gemessenen Zahlen aus diesem
Projekt als Beleg. Gedacht als **Ausgangspunkt für eigene Recherche**, nicht
zum direkten Übernehmen — zu jedem Abschnitt stehen deshalb die Fachbegriffe
dabei, unter denen sich Literatur finden lässt.

Stand 14.08.2026.

---

## 1. Wie das Lernen tatsächlich abläuft

Es gibt **ein** neuronales Netz. Kein Wettbewerb, keine Generationen, keine
Auswahl des Besten.

```
Auto fährt mit dem aktuellen Netz
  → Bild rein, Aktion raus, Reward zurück
  → Erfahrung wird gespeichert
  → regelmäßig werden die Gewichte per Gradientenabstieg verschoben
  → dasselbe Netz, kontinuierlich verändert
```

Lauf `1432` machte dabei **997.499 Gewichtsanpassungen**.

`best_model.zip` ist **keine** Grundlage für den weiteren Verlauf, sondern eine
Momentaufnahme, die bei einer besonders guten Zwischenauswertung abgelegt wird.
Das Training läuft mit den aktuellen Gewichten weiter, auch wenn die gerade
schlechter sind. Deshalb konnte Lauf `2331` nach seinem Zusammenbruch bei
250.000 Steps nicht auf sein bestes Modell zurückfallen.

**Abgrenzung zu evolutionären Verfahren:** Diese hätten eine Population,
bewerteten jedes Mitglied über die Gesamtsumme einer Episode und erzeugten
daraus die nächste Generation. Sie brauchen keinen Gradienten, kein `gamma`,
keine Wertfunktion — genau die Fehlerquellen, an denen dieses Projekt
gescheitert ist, hätten sie nicht. Sie skalieren aber schlecht mit der
Parameterzahl: die Vision-Policy hier hat **16.261.098 Parameter** (davon
5.288.614 im Actor), das ist für Evolution Strategies auf einem Rechner
aussichtslos. Der Lidar-Agent mit rund 10.000 Parametern wäre dagegen gut
machbar.

*Suchbegriffe:* policy gradient, actor-critic, evolution strategies,
CMA-ES, OpenAI ES, neuroevolution

---

## 2. SAC gegen PPO: der strukturelle Unterschied

Der Unterschied liegt darin, **was mit gesammelter Erfahrung passiert**.

### SAC — off-policy, mit Gedächtnis

```
buffer_size    500.000   Speicher für alte Erfahrungen
train_freq     2         alle 2 Schritte wird gelernt
gradient_steps 1         dann ein Gradientenschritt
batch_size     512       auf 512 zufällig gezogenen alten Erfahrungen
```

Eine Situation von vor Stunden kann noch hundertfach zum Lernen dienen.

### PPO — on-policy, ohne Gedächtnis

```
n_steps    512    pro Env sammeln
n_envs      12    → 512 × 12 = 6.144 Frames pro Runde
n_epochs    10    diese 6.144 zehnmal durchgehen
batch_size 256    in Minibatches
→ 240 Gradientenschritte, dann wird alles verworfen
```

PPO darf nur mit Erfahrung lernen, die die **aktuelle** Politik erzeugt hat.
Sobald sich die Gewichte ändern, sind die alten Daten ungültig.

### In Zahlen

Pro 6.144 gesammelten Frames macht SAC rund **3.072** Gradientenschritte,
PPO **240**. Faktor 13 — und SAC verwendet jede Erfahrung zusätzlich mehrfach.

**Konsequenz für die Wahl:** Wo Samples teuer sind (eine simulierte,
gerenderte Umgebung, ein Env), gewinnt SAC. Wo Samples billig sind (tausend
parallele Envs), verschiebt sich das zugunsten von PPO, und SACs Replay-Buffer
wird zum Speicherproblem.

*Suchbegriffe:* off-policy vs on-policy, sample efficiency, replay ratio,
update-to-data ratio (UTD), trust region, importance sampling

---

## 3. Der Replay-Buffer

Ein Ringspeicher vergangener Übergänge:

```
(Beobachtung, Aktion, Reward, nächste Beobachtung, fertig?)
```

`buffer_size = 500.000` bedeutet die letzten 500.000 Schritte des Agenten. Bei
30 fps sind das **4.6 Stunden simulierte Fahrzeit** oder 250 volle Episoden.
Ist er voll, wird der älteste Eintrag überschrieben.

### Drei Aufgaben

1. **Wiederverwendung** — jeder Übergang wird hunderte Male gezogen
2. **Entkopplung** — aufeinanderfolgende Frames sind fast identisch; zufälliges
   Ziehen bricht die zeitliche Korrelation, sonst lernt das Netz aus 512
   nahezu gleichen Situationen
3. **Gegen das Vergessen** — der Critic soll auch Zustände bewerten können, die
   die aktuelle Politik nicht mehr ansteuert

### Speicherbedarf

Der Buffer speichert **Rohbeobachtungen**. Bei Bildern wird das schnell brutal:

```
3 gestapelte Bilder à 100×166 uint8 = 49.800 Bytes je Übergang
× 500.000                            = 23.2 GB
```

Zum Vergleich: dieselben 500.000 Übergänge mit der Lidar-Beobachtung
(4 float32) belegen **7.6 MB**. Faktor 3000.

`optimize_memory_usage=True` halbiert das, indem `next_observations` nicht
separat abgelegt werden. Ohne die Option waren es 46.4 GB — Lauf `2246` lief so
und brauchte 10 Stunden für nur 720.000 Steps.

*Suchbegriffe:* experience replay, prioritized experience replay,
catastrophic forgetting, replay buffer memory optimization

---

## 4. Diskontfaktor gamma — der teuerste Fehler dieses Projekts

`gamma` bestimmt, wie weit der Agent in die Zukunft plant:

```
Horizont ≈ 1 / (1 - gamma)

gamma 0.99  → 100 Frames  = 3.3 s bei 30 fps
gamma 0.995 → 200 Frames  = 6.7 s
gamma 0.999 → 1000 Frames = 33 s
```

### Der Befund

Das Auto braucht aus 1.3 m/s **1.9 Sekunden und 97 Frames** zum Stehen. Der
Horizont bei `gamma = 0.99` beträgt 100 Frames. Der Agent operierte also exakt
an der Grenze dessen, was er überhaupt vorausschauen konnte — und crashte,
obwohl er sieben Frames lang voll bremste.

### Warum die Korrektur trotzdem scheiterte

`gamma` von 0.99 auf 0.999 zerlegte das Training: Zusammenbruch nach 250.000
Steps, danach 1.25 Millionen Steps flache Linie bei Reward −355.

Ursache ist die **Wertskala**. Die zu lernende Wertfunktion wächst mit
`1/(1-gamma)`:

```
bei +2.9 Reward pro Frame:
gamma 0.99  → Wertfunktion bis ca.  290
gamma 0.999 → Wertfunktion bis ca. 2900
```

Der Critic muss also zehnmal größere Zahlen vorhersagen — bei unveränderter
Lernrate von 5e-5. Wer `gamma` deutlich anhebt, muss die Lernrate, die
Reward-Skalierung oder beides mitdenken.

**Für die Arbeit relevant:** Das ist ein sauber dokumentierter Fehlschlag mit
klarer Ursache. Solche negativen Ergebnisse sind berichtenswert.

*Suchbegriffe:* discount factor, effective horizon, value function scaling,
reward scaling, Pop-Art normalization, credit assignment

---

## 5. Lernrate

`5e-5` = `0.00005`, die Schrittweite bei jedem Gradientenschritt. Der Gradient
gibt nur die Richtung an, die Lernrate wie weit man geht.

```
zu groß  → überschießt, schwingt, entgleist
zu klein → quälend langsam, bleibt in schlechten Lösungen hängen
```

In diesem Projekt: SAC durchgehend `5e-5`, PPO `3e-4` und `1e-4`. Das ist
üblich — PPO verträgt größere Schritte, weil es über einen ganzen Datenblock
mittelt und mit `clip_range` einen eingebauten Begrenzer hat.

### Wechselwirkung mit vortrainierten Gewichten

Die aus dem Vortraining kopierten CNN-Gewichte werden **nicht eingefroren**.
Eine hohe Lernrate kann das mühsam antrainierte Sehen in den ersten Updates
wieder zerstören. Genau das ist die naheliegende Erklärung dafür, dass der
einzige PPO-Lauf mit Backbone (`1601`, lr 3e-4) nach **neun Frames** crashte.

Ungetestet, weil es keinen PPO-Lauf mit Backbone **und** niedriger Lernrate
gibt — die letzte leere Zelle im Versuchsplan.

*Suchbegriffe:* learning rate schedule, catastrophic forgetting in transfer
learning, layer freezing, discriminative fine-tuning

---

## 6. Framestapel — wofür er da ist und wann er versagt

Ein Einzelbild zeigt, **wo** das Auto ist, nicht **wie schnell**. Der Stapel
aus drei aufeinanderfolgenden Bildern soll dem Agenten erlauben, Geschwindigkeit
und Drehrate aus den Unterschieden abzuleiten.

### Wann das nicht funktioniert

```
Beobachtung 166×100, Maßstab 47.2 px/m
Bewegung pro Frame:
   0.10 m/s → 0.16 px
   0.30 m/s → 0.47 px
   0.50 m/s → 0.79 px
   0.80 m/s → 1.26 px

Erst ab 0.64 m/s bewegt sich das Auto um mindestens einen Pixel.
```

Unterhalb davon enthalten die drei Bilder **dieselbe Information dreimal**.
Der Agent verbringt 33 % seiner Zeit dort, und die gesamte Anfahrphase
(2.8 s bis 0.6 m/s) liegt in diesem Bereich.

### Historische Einordnung

Der Test „3 Bilder besser als 1" wurde am 08.06. unter der **alten** Physik
gemacht, wo das Auto sich mit 2.0 px pro Frame bewegte. Damals funktionierte
der Stapel. Die Umstellung auf das gemessene Fahrzeugmodell hat ihm die
Grundlage entzogen, ohne dass der Test wiederholt wurde.

### Die Alternative

Geschwindigkeit und Lenkwinkel **direkt** als Zahlen mitgeben. Im Einspurmodell
gilt

```
omega = v · tan(current_steer · max_steer_angle) / L
```

Die Drehrate ist also eine reine Funktion von `v` und `current_steer`. Wer
beide exakt bekommt, hat alles, was der Stapel über Bewegung sagen könnte —
für 8 Bytes statt 33.200.

*Suchbegriffe:* frame stacking, partial observability, POMDP, velocity
estimation from pixels, proprioception, asymmetric actor-critic

---

## 7. Auflösung der Beobachtung

```
Streckenbild        830 × 500 px
Beobachtung         166 × 100 px   (5-fach verkleinert)
Maßstab              47.2 px/m

Das Auto darin:     4.6 × 1.8 Pixel  ≈ 8 Pixel gesamt
```

### Folge: die Policy ist extrem spröde

Zwischen dem cv2- und dem pygame-Renderpfad unterscheiden sich **0.05 % der
Pixel** — reine Kantenglättung des Auto-Sprites. Das reicht, um die Aktionen
vollständig zu ändern:

```
hidden-Bild : Gas 0.488  Bremse 0.974  Lenk +0.302
human-Bild  : Gas 0.875  Bremse 0.908  Lenk −0.801
```

Kein Rätsel, wenn man die Größenordnung kennt: die acht abweichenden Pixel
entsprechen ungefähr **dem gesamten Fahrzeug** in der Beobachtung. Aus Sicht
des Netzes war das Auto weg.

Praktische Auswirkung: derselbe Agent fuhr im Trainings-Renderpfad 256 Frames
und 6.9 m, im anderen **0.00 m** — er stand und lief nach 750 Frames in den
Reward-Boden.

**Für die Übertragung auf echte Hardware ist das vernichtend.** Wenn schon die
Antialiasing-Kante des eigenen Renderers ausreicht, hat eine echte Kamera mit
Beleuchtungsschwankungen, Schatten und Rauschen keine Chance.

### Gegenmittel

Höhere Auflösung, mehr Graustufen (kostet nichts, siehe Abschnitt 9), und vor
allem **Augmentierung der Beobachtung** beim Training — leichtes Rauschen,
Verschiebungen um wenige Pixel, Helligkeitsschwankungen. Fehlt in diesem
Projekt bisher vollständig.

*Suchbegriffe:* data augmentation for RL, DrQ, RAD, domain randomization,
sim-to-real transfer, observation robustness

---

## 8. Was Behaviour Cloning tatsächlich überträgt

Die Kette im Projekt:

```
1. Lidar-Agent (PPO, 4 Eingangswerte)   ← fährt gut: 11 Runden, kein Crash
2. zeichnet 12.000 Paare (Bild → Aktion) auf
3. CNN lernt überwacht, diese Aktionen aus Bildern vorherzusagen
4. davon werden NUR die 3 Faltungsschichten in den RL-Agenten kopiert
```

```
Conv 8×8 → Conv 4×4 → Conv 3×3 → Flatten → Linear → Linear
└──────── wird kopiert ─────────┘          └── verworfen ──┘
```

Übertragen wird also **nur Sehfähigkeit**, kein Fahrwissen. Der Agent erbt
„so sieht ein Streckenrand aus", nicht „und deshalb bremst man hier". Die
Reward-Regeln kennt das Vortraining ohnehin nicht — es hat nie einen Reward
gesehen.

Zusätzlich werden die kopierten Gewichte **nicht eingefroren** und können im
Lauf von Millionen Schritten wieder wegoptimiert werden.

### Was mehr bringen würde

1. Faltungsschichten anfangs einfrieren
2. auch die Linear-Schichten übernehmen, Actor damit initialisieren
3. **Replay-Buffer mit Experten-Übergängen vorfüllen** — SAC lernt off-policy,
   kann also direkt aus fremden Übergängen lernen. Dafür müsste der Experte
   neu aufgezeichnet werden, mit vollständigen Übergängen
   (`obs, action, reward, next_obs, done`) statt nur Bild und Aktion.

*Suchbegriffe:* behaviour cloning, imitation learning, cross-modal distillation,
DDPGfD, SACfD, learning from demonstrations, DAgger

---

## 9. Speicherformat: Redundanz und Kompression

### Der Framestapel wird dreifach gespeichert

SB3 legt die **fertig gestapelte** Beobachtung im Buffer ab, weil
`VecFrameStack` die Umgebung umhüllt. Benachbarte Übergänge teilen sich aber
zwei ihrer drei Bilder — jedes Einzelbild liegt also rund dreimal im Speicher.

```
                              gestapelt    einzeln gespeichert
166×100, n_stack=3              23.2 GB              7.7 GB
250×150, n_stack=3              52.4 GB             17.5 GB
300×180, n_stack=3              75.4 GB             25.1 GB
```

Umsetzung erfordert einen eigenen Replay-Buffer, der beim Speichern nur das
neueste Bild ablegt und beim Ziehen die Stapel aus Nachbarindizes zusammensetzt.
Die Feinheiten sind **Episodengrenzen** und der **Umlauf des Rings** (13c) —
sonst stapelt man über einen `reset()` oder über die Naht hinweg. So arbeiten
die DQN-Implementierungen von DeepMind, Dopamine und rlpyt; SB3 hat es nur nicht
eingebaut.

### Was es kostet — und wie man sich dabei vermisst

Naheliegende Annahme: beim Ziehen wird dieselbe Datenmenge bewegt, nur über
drei Indizes statt einen, also ist die Ersparnis kostenlos.

**Die Annahme ist falsch.** Gemessen an der echten Konfiguration:

```
Standard-Buffer     49.1 ms/Update     2849.6 MB
Einzelbilder        54.9 ms/Update      949.9 MB
                    +11.8 %            Faktor 3.00 weniger
```

Der Aufschlag steckt vollständig im Ziehen: 8.6 → 13.9 ms je Batch. Die Bytes
sind zwar dieselben, aber der Stapel **existiert noch nicht** und muss erst
entstehen — ein zusätzlicher Schreibdurchgang über die volle Batchgröße.

Zwei methodische Punkte, die dabei mehr wert sind als das Ergebnis:

1. **Die erste Messung war irreführend**, weil sie mit `batch_size=32` lief
   statt mit den echten 512. Dort ergab sich 12.9 gegen 12.8 s, also
   scheinbare Kostenfreiheit. Ein Mikro-Benchmark muss bei der Losgröße laufen,
   die später auch benutzt wird — sonst dominieren feste Kosten.

2. **Die naive Umsetzung war dreimal so teuer** wie nötig (26.1 ms statt 8.6).
   Ursache war nicht der Datenzugriff — ein einzelner Gather von 512 Bildern
   kostet 1 ms — sondern das stückweise Zusammensetzen mit `np.concatenate`.
   Ein einziger Zugriff über ein Indexfeld der Form (N, n_stack) liefert das
   Ergebnis bereits in der richtigen Speicherreihenfolge, die anschließende
   Umformung ist dann kostenlos. Damit 13.9 ms.

Ob sich der Tausch lohnt, hängt davon ab, was knapp ist. Hier war der
Arbeitsspeicher die bindende Grenze für Auflösung und Buffergröße, die
Rechenzeit nicht — also ja. Bei gleicher Wanduhrzeit bedeutet er allerdings
rund 10 % weniger Gradientenschritte, was beim Vergleich von Läufen über diese
Änderung hinweg zu berücksichtigen ist.

### Bit-Packing lohnt sich nicht

Bei vier Graustufen braucht ein Pixel nur 2 Bit. Vier Pixel in ein Byte gepackt
spart Faktor 4 — gemessen und verlustfrei:

```
gepackt      : 5.8 GB statt 23.2 GB
entpacken    : +28.8 ms bei JEDEM Gradientenschritt
               = +59 % auf die 48.5 ms pro Update
```

Aus 12 Stunden Training würden 19. Speicher gegen Zeit — und Zeit ist hier das
knappere Gut.

### Graustufen kosten nichts

Ein häufiges Missverständnis: mehr Graustufen erhöhen den Aufwand **nicht**.
Das Bild ist `uint8`, ein Byte pro Pixel, egal ob vier oder 256 Werte darin
vorkommen. Die Quantisierung wirft nur Information weg, sie spart keinen
Speicher.

In diesem Projekt steht `levels = 4`, umgesetzt als

```python
np.floor_divide(bild, 256 // levels) * (256 // levels)     # virtual_camera.py
```

Die Bilder enthalten also nur die Werte 0, 64, 128, 192 — bei 8 Bit Platz pro
Pixel. Der Schritt auf 256 Stufen wäre in Speicher und Rechenzeit **exakt
gratis**; das Byte wird ohnehin geschrieben.

Gratis ist er allerdings nur technisch. Der Agent bekommt dann einen anderen
Eingang, und das vortrainierte Backbone hat auf vier Stufen gelernt (siehe 8).
Es ist also kein kostenloses Extra, sondern ein eigenes Experiment.

*Suchbegriffe:* frame stacking in replay buffer, lazy frames, memory-efficient
replay, quantization

---

## 10. Fallen in der Reward-Gestaltung

Vier Muster, die in diesem Projekt tatsächlich aufgetreten sind.

### Die Sperrklinke

Jede Runde wurde gegen die bisherige Bestzeit der Episode gerechnet — **jedes
Mal aufs Neue**. Eine einzelne schnelle Runde hob damit die Messlatte und
kostete über den Rest der Episode mehr, als sie einbrachte:

```
sieben Runden à 300 Frames                      →     0
davon eine mit 250 Frames                       →  −250
stetig schneller, 300 → 240                     →  +300
```

**Schnellfahren war unterm Strich bestraft.** Die Boni teleskopierten (in Summe
immer `erste minus beste Runde`), die Strafen nicht — sie fielen bei jeder
Runde erneut an.

### Die tote Zone

Eine harte Strafe unterhalb von 0.1 m/s und eine quadratisch wachsende
Belohnung darüber ließen den Bereich dazwischen leer:

```
v = 0.15 m/s → Strafe 0.0, Reward 0.045/Frame
v = 0.30 m/s → Strafe 0.0, Reward 0.180/Frame
```

Genau dort landet das Auto beim Bremsen in Kurven. Ein lineares Gefälle statt
der Klippe schließt die Lücke.

### Absolute Strafen gegen skalierende Belohnungen

Der Speed-Reward wächst mit `v²`, die Lenkstrafen sind konstant. Der
Gleichstand liegt bei

```
2v² = 0.58  →  v = 0.54 m/s
```

Unterhalb davon kostet Lenken mehr, als Fahren einbringt — ein langsamer Agent
hat den Anreiz, **weniger zu lenken**, und fährt geradeaus in die Wand.

### Strafen, die gutes Verhalten treffen

Der Oszillationsschutz sollte Zappeln bestrafen, besteuerte aber jedes Lenken:
der gut fahrende Lidar-Experte zahlte **−625 über 2000 Frames**, mehr als seine
gesamten Rundenboni (+600). Die Schwelle war auf 2.0 gesetzt, sein Mittelwert
lag bei 5.9.

### Und ein Log-Problem

`best_lap_bonus` (200.0) und `slow_lap_penalty` (−30.0) standen in jeder
Konfiguration und wurden **nie gelesen** — die echten Faktoren waren
hartcodierte 5.0 und 2.0. Wer die JSON las, betrachtete Zahlen ohne Wirkung.

*Suchbegriffe:* reward shaping, potential-based reward shaping, reward hacking,
specification gaming, sparse vs dense rewards

---

## 11. Messmethodik: Stolpersteine

### `eval_freq` wird pro Env gezählt

SB3 zählt `eval_freq` **pro Environment**. Aus `50000` bei 12 Envs wird eine
Messung alle 600.000 Schritte, während SAC mit einem Env alle 10.000 misst —
60-fach seltener. Weil `best_model` nur bei einer Auswertung geschrieben wird,
ist dessen Auswahl entsprechend gröber und der berichtete Bestwert eine
**untere Schranke**.

Das Training selbst beeinflusst die Auswertungsdichte nicht. Der `EvalCallback`
greift nicht in die Gewichte ein — es ändert sich nur, was man sieht.

### `gradient_steps` muss mit `n_envs` mitwachsen

Bei `train_freq=2, gradient_steps=1` und 3 Envs sammelt ein Aufruf drei
Transitionen — also ein Gradientenschritt pro sechs statt pro zwei. Lauf `2347`
hatte dadurch **249.166 statt 607.499** Updates bei gleicher Step-Zahl.

Der Lauf galt lange als Beleg dafür, dass ein kleinerer Replay-Buffer schadet.
Die Gegenrechnung entkräftet das:

```
2347 mit 249.166 Updates            → bestes 144.4
2139 beim selben Update-Stand       → bestes 133.6
```

Praktisch identisch. **Der Buffer war nie das Problem**, der Lauf war schlicht
untertrainiert.

### Deterministische Auswertung

Umgebung deterministisch, `deterministic=True`, fester Startzustand → die fünf
Eval-Episoden sind bei sieben von neun Läufen zu **100 % identisch**. Jeder
Punkt der Lernkurve ist damit **eine** Episode, kein Mittelwert. Der Spitzenwert
3828 ist ein einzelner Durchlauf.

### Spitze und Ende gehören beide berichtet

Jeder SAC-Lauf erreicht einen Höhepunkt und fällt danach ab:

```
1432:  3828 → −1421
2139:  2428 →   546
2210:   876 →  −316
2246:   633 →   250
```

Nur den Peak zu nennen überzeichnet, nur das Ende unterschlägt das Erreichbare.

### Zeit oder Samples als Vergleichsmaßstab

Beides ist legitim, beantwortet aber Verschiedenes:

- **Gleiche Step-Zahl** → Sampleeffizienz. Benachteiligt PPO strukturell, weil
  es jede Erfahrung nach einem Update verwirft.
- **Gleiche Rechenzeit** → praktischer Nutzen bei gegebenem Budget.

Wer zeitlich budgetiert und die Step-Zahl mitprotokolliert, bekommt aus einem
einzigen Lauf beide Aussagen.

Nicht vergessen: die Auswertung selbst kostet Rechenzeit. SAC verbrauchte
**11 % seiner Schritte** für Evaluation (150 Auswertungen × 5 Episoden =
167.905 Frames), PPO praktisch nichts.

*Suchbegriffe:* evaluation protocol in RL, statistical significance in deep RL,
"Deep RL that Matters", rliable, seed variance

---

## 12. Physik und ihre Rückwirkung auf alles andere

Die Umstellung von einem geschätzten auf ein gemessenes Fahrzeugmodell hat
gezeigt, wie viele Einstellungen implizit an der Fahrzeugdynamik hängen.

```
alt: Beschleunigung 12.5 m/s²,  Bremsen 14.4 m/s²
neu: Beschleunigung  0.25 m/s², Bremsen  0.68 m/s²
```

Alle Zeitkonstanten waren rund **20-fach zu kurz** — Ursache war eine viel zu
kleine Masse (0.080 kg) gegenüber den angesetzten Kräften. Die
Endgeschwindigkeit stimmte dagegen ungefähr.

Was daran anschließend nicht mehr passte:

| Einstellung | Warum sie nicht mehr passte |
|---|---|
| `gamma = 0.99` | Bremsweg 97 Frames gegen 100 Frames Horizont |
| Framestapel | Bewegung unter 0.64 m/s ist subpixelig |
| `total_timesteps` | Lidar-Experte brauchte 500k statt 80k Steps |
| Standstill-Strafe | Anfahren dauert 12 statt 1 Frame |
| fahrzeugfeste Ansicht | Bremsweg länger als die Sichtweite (siehe unten) |
| `max_steer_change` | 1.0 hieß Anschlag zu Anschlag in 67 ms |

### Die fahrzeugfeste Ansicht wurde dadurch unbrauchbar

`camera_view="crop"` schneidet 150×150 px um das Auto und skaliert auf 84×84.
Das Auto wäre darin **12.9 × 5.0 px** statt 4.6 × 1.8 — achtmal so viele Pixel,
und die Subpixelgrenze sänke von 0.64 auf 0.23 m/s.

Der Ausschnitt zeigt aber nur 0.64 m Kantenlänge, also 0.32 m nach vorn — 8 %
einer Runde. Gegen den gemessenen Bremsweg:

```
aus 0.5 m/s: 0.38 m Bremsweg   sichtbar 0.32 m   zu kurz
aus 0.8 m/s: 0.81 m            sichtbar 0.32 m   zu kurz
aus 1.2 m/s: 1.52 m            sichtbar 0.32 m   zu kurz
aus 1.6 m/s: 2.33 m            sichtbar 0.32 m   zu kurz
```

Der Agent sähe die Wand grundsätzlich erst, wenn Bremsen nicht mehr reicht —
bei jedem Tempo. Mit dem alten Auto (Bremsweg 6 cm bei 1.5 m/s) war die Ansicht
brauchbar, deshalb funktionierte sie im Juni.

Das Grundproblem ist strukturell: mehr Pixel fürs Auto bedeuten bei einem
mitgeführten Ausschnitt zwangsläufig weniger Sichtweite. Beides zugleich geht
nur über die Auflösung.

**Merksatz:** Wer die Fahrzeugdynamik ändert, muss jede Einstellung prüfen, die
auf Zeitskalen beruht — Diskontfaktor, Episodenlänge, Framestapel, Zeitlimits,
Strafschwellen.

*Suchbegriffe:* system identification, sim-to-real gap, domain adaptation,
time discretization in RL, action repeat / frame skip

---

## 12b. Die Videomessung: was sie belegt und was nicht

Das Fahrzeugmodell stammt aus Videos (30 fps, Kalibrierung 14 px = 35 mm).
Drei Prüfungen entscheiden darüber, wie belastbar es ist.

### Perspektive ausgeschlossen

Wenn die Kamera schräg auf die Strecke schaut, ändert sich der Maßstab über die
Bildhöhe — dieselbe reale Bewegung erzeugt unten mehr Pixel als oben. Das würde
eine Beschleunigung **vortäuschen**, die es nicht gibt.

Prüfbar, weil das Auto eine feste Länge hat. Über die volle Bildhöhe gemessen:

```
y    0-150 px : 44.0 px Fahrzeuglänge   (Anschnitt am oberen Rand)
y  150-300 px : 49.3 px
y  450-600 px : 49.2 px
y  900-1100 px: 49.4 px
```

Konstant. Die Kamera schaut senkrecht, `PX_PER_M = 400` gilt über das ganze
Bild. Ohne diese Prüfung wäre der gemessene Beschleunigungsverlauf wertlos
gewesen.

### Ein Messvideo ist unbrauchbar

Dieselbe Prüfung deckte auf, dass in `Beschleunigung mit Speed V2` etwas
anderes mit im Blob steckt:

```
y    0-200 px : 134 px Länge
y  200-400 px :  50 px
y  600-800 px : 297 px
y  800-1100 px: 691 px      ← ein 48-px-Auto kann das nicht sein
```

Der Median wandert damit weg von der Fahrzeugmitte. Aus diesem Video stammt der
Wert **a = 0.313 m/s²**, der als zweiter Stützpunkt in die Antriebskennlinie
eingeht. Er ist mit Vorsicht zu behandeln — die 44 % Abweichung zum anderen
Durchfahrt-Video (0.488) könnte daher rühren.

Ein Längenfilter (Blob verwerfen, wenn er stark vom Median abweicht) wurde
implementiert und wieder zurückgenommen, um die bereits ausgewerteten Zahlen
nicht zu verändern. Für eine Neuauswertung wäre er einzubauen.

### v_max ist nicht bestimmbar

Das physikalisch richtige Modell `dv/dt = (v_inf − v)/T` wurde gefittet und
mittels Profil-Test geprüft, ob `v_inf` durch die Daten überhaupt festgelegt
ist:

```
 v_inf   tau_fit   RMSE(cm)        Messfenster: 3.57 s
  1.40     3.31      4.07
  1.80     4.69      3.04
  3.00     8.79      1.83
  6.00    18.97      1.10
```

Der Fehler sinkt **monoton** — es gibt kein Minimum. Die Daten bevorzugen den
Grenzfall `tau → ∞`, also konstante Beschleunigung. Das Auto erreicht im
Bildausschnitt nie die Sättigung.

**Konsequenz:** `v_max = 1.90 m/s` im Modell ist eine Annahme, keine Messung.
Belegt ist nur, dass 1.844 m/s erreicht wurden. Ebenso ist der Rollwiderstand
`R0 = 0.05 m/s²` geschätzt — es gibt kein Ausroll-Video. Beide Werte sind im
Code als Annahme gekennzeichnet.

Sauber gemessen sind dagegen: die Bremsverzögerung (`a(v) = 0.217 + 0.336·v`,
trifft beide Bremsversuche exakt, RMSE 0.4–0.6 cm über 2.5 m Fahrstrecke) und
der haftungsbegrenzte Ast der Beschleunigung.

### Methodische Punkte der Auswertung selbst

Vier Fehler, die erst nach mehreren Anläufen gefunden wurden:

1. **Skalierungsfehler im Ableitungsfilter.** Das Resampling-Gitter hatte einen
   anderen Abstand als das an den Savitzky-Golay-Filter übergebene `delta` —
   alle Geschwindigkeiten lagen 2–4 % zu hoch.
2. **Angeschnittene Frames.** Solange das Auto am Bildrand hängt, *wächst* der
   Blob statt sich zu bewegen; sein Median wandert halb so schnell wie das
   Fahrzeug. Eine Pixelzahl-Heuristik lässt genau die Übergangsframes durch —
   nötig ist die geometrische Prüfung, ob Ober- **und** Unterkante im Bild
   liegen.
3. **Kennwerte aus Positionsfits, nicht aus Ableitungen.** Die Suche nach dem
   Maximum eines verrauschten Geschwindigkeitssignals ist nach oben verzerrt,
   und der Fehler wandert direkt in die Bremsverzögerung.
4. **Der Zeitnullpunkt beim Anfahren** muss über die Blob-Unterkante bestimmt
   werden, nicht über den Median — die Unterkante bewegt sich mit der echten
   Fahrzeuggeschwindigkeit, auch wenn oben abgeschnitten. RMSE dadurch von
   2.4 cm auf 0.5 cm.

*Suchbegriffe:* system identification, optical flow, Savitzky-Golay filter,
parameter identifiability, profile likelihood

---

## 12c. Werkzeug-Fallen, die Messungen verfälschen

### `DummyVecEnv` setzt automatisch zurück

Endet eine Episode, ruft `DummyVecEnv` sofort `reset()` auf — **bevor** die
eigene Schleife `render()` erreicht. Der Crash-Frame wird nie gezeichnet; was
man sieht, ist bereits der Startzustand.

Das führte hier zu einer völlig falschen Diagnose: das Auto schien einfach
stehenzubleiben und neu zu starten. Tatsächlich fuhr es mit 1.19 m/s in die
Außenwand und bremste die letzten sieben Frames vergeblich.

Wer das Verhalten am Episodenende beobachten will, braucht entweder die rohe
Umgebung ohne Vec-Wrapper oder muss den Zustand vor dem Schritt zwischenspeichern.

### Renderpfad und Trainingspfad müssen übereinstimmen

Die Beobachtungszelle nutzte `render_mode="human"`, trainiert wurde mit
`"hidden"`. Der Unterschied beträgt 0.05 % der Pixel — genug, um das Verhalten
vollständig zu ändern (siehe Abschnitt 7). Man sieht dann nicht den Agenten,
den man trainiert hat.

### `deterministic=False` bei der Beobachtung

Die Beobachtungszelle zog aus der Policy-Verteilung, die Auswertung im Training
nutzt `deterministic=True`. Kein Fehler, aber die beiden Bilder sind nicht
dieselben.

---

## 13. Reproduzierbarkeit

Das durchgehende Muster in diesem Projekt: **ein Wert steht im Code und wird
getrennt davon von Hand ins Log geschrieben.** Beides läuft auseinander.

Aufgetretene Fälle:

- die drei Koeffizienten der Längsdynamik waren hartcodiert und tauchten in
  keiner Konfiguration auf — `mass_kg` wurde dagegen geloggt und täuschte
  Vollständigkeit vor
- `n_envs` stand auf 1, SB3s Modellarchiv nannte 3
- `total_timesteps` stand zweimal: Log 1.5M, gelaufen 2.5M
- `gamma` war gar nicht gesetzt, SB3 nahm still 0.99
- `best_lap_bonus` und `slow_lap_penalty` wurden geloggt, aber nie gelesen

**Die Gegenmittel:**

1. Werte aus dem Objekt lesen, nicht aus einer Variablen wiederholen
   (`vec_env.num_envs` statt `n_envs`)
2. jede Zahl nur an einer Stelle definieren
3. **Git-Commit mitloggen** — damit lässt sich der exakte Code wiederherstellen,
   unabhängig davon, ob eine Einstellung im Log auftaucht. Dazu gehört ein
   `dirty`-Flag: bei unversionierten Änderungen beschreibt der Hash den Lauf
   nur unvollständig.
4. eine Versionskennung für die Reward-**Formel**, nicht nur für ihre Gewichte

*Suchbegriffe:* experiment tracking, reproducibility in machine learning,
MLflow, Weights & Biases, configuration management, hydra

---

## 13b. Episodenende: ein Zeitlimit ist kein Endzustand

Der theoretisch sauberste Punkt dieses Projekts — und der, der am längsten
unbemerkt falsch war.

### Zwei Arten zu enden

Eine Episode kann aus zwei grundverschiedenen Gründen aufhören:

| | was passiert | Wert danach |
|---|---|---|
| **terminated** | Crash. Der Zustand ist absorbierend. | tatsächlich 0 |
| **truncated** | Frame 2000 erreicht. Wir hören auf zu schauen. | *nicht* 0 — es ginge weiter |

Für das Lernziel des Critics macht das den Unterschied zwischen

```
terminated:   y = r
truncated:    y = r + gamma · Q(s', a')
```

Beim echten Ende ist das Abschneiden richtig. Beim Zeitlimit ist es eine
Behauptung über die Welt, die schlicht falsch ist: das Auto wäre
weitergefahren, wir haben nur die Aufzeichnung beendet. Das Zeitlimit ist eine
Eigenschaft **unseres Versuchsaufbaus**, keine Eigenschaft der Aufgabe.

### Wie es in SB3 schiefging

SB3 unterscheidet beides über `handle_timeout_termination`. Der Buffer merkt
sich dann in `timeouts` getrennt mit und rechnet beim Ziehen

```python
dones * (1 - timeouts)
```

Nachgewiesen:

```
handle_timeout_termination=True    Crash done=1,  Zeitlimit done=0
handle_timeout_termination=False   Crash done=1,  Zeitlimit done=1
```

In diesem Projekt stand es auf `False` — nicht absichtlich, sondern weil
`optimize_memory_usage=True` es erzwingt. Beide Flags sind in SB3 unvereinbar,
und die Speicheroptimierung war gewollt. Der Lernfehler kam als
**stillschweigende Nebenwirkung einer Speicherentscheidung** herein. Es gibt
keine Warnung.

### Warum der Schaden nicht lokal bleibt

Naheliegender Einwand: betroffen sind doch nur die letzten Frames einer
Episode, bei gamma=0.99 also die letzten ein- bis dreihundert von 2000. Das
wäre verkraftbar.

Der Einwand greift nicht, und der Grund ist die Beobachtung selbst:

> **Der Agent sieht keine Uhr.** Seine Beobachtung ist ein Kamerabild. Weder
> Framezähler noch verbleibende Zeit sind darin enthalten.

Für einen Agenten, der saubere Runden fährt, sieht Frame 1990 **identisch aus**
wie Frame 300 — dieselbe Stelle der Strecke, dieselbe Lage des Autos, dasselbe
Bild. Der Critic bekommt für denselben Eingang widersprüchliche Ziele:

```
bei Frame  300  →  y = r + gamma · Q(s')     "viel wert, es geht weiter"
bei Frame 1990  →  y = r                     "wertlos, hier endet die Welt"
```

Er kann die Fälle nicht trennen und mittelt sie. Der falsche Nullwert klebt
also nicht am Episodenende, sondern **verschmiert über alle Zustände, die gut
aussehen**. Aus einem lokalen Randfehler wird eine globale Verzerrung der
Wertfunktion.

In der Literatur ist das der Grund, warum bei endlichen Zeitlimits entweder
korrekt gebootstrappt oder die verbleibende Zeit in die Beobachtung
aufgenommen wird (*time-awareness*). Beides fehlte hier.

### Warum es ausgerechnet die guten Läufe trifft

```
schlechter Agent  →  crasht früh   →  erreicht 2000 nie     →  kein Schaden
guter Agent       →  fährt durch   →  erreicht 2000 immer   →  volle Wirkung
```

Der Schaden ist **positiv mit der Leistung korreliert**. Er schaltet sich genau
in dem Moment ein, in dem der Agent aufhört zu crashen, und trifft ausgerechnet
die Zustände des gelungenen Fahrens.

Lauf `1432` hatte auf seinem Höhepunkt eine mittlere Episodenlänge von **exakt
2000** — jede Episode lief also ins Limit. Danach fiel er von 3828 auf −1421.

**Status: begründete Hypothese, nicht bewiesen.** Der Mechanismus ist
zwingend, seine Größenordnung nicht gemessen. SAC hat weitere bekannte
Zusammenbruchsarten (davonlaufende Q-Werte, kippende Entropie-Regelung), die
dasselbe Muster erzeugen können. Der nächste Referenzlauf prüft es.

### Ist das Limit überhaupt sinnvoll gewählt?

Die Zeile stammt aus dem ersten funktionierenden PPO-Commit (18.05.2026), ohne
Kommentar, nie überarbeitet — und damit aus der Zeit **vor** der
Physikkorrektur, also für ein rund 20-fach zu schnelles Fahrzeug.

Nachgerechnet unter `measured_v2`:

```
Rundenlänge          7.66 m   (1807 px Mittellinie bei 236 px/m)
Zeitlimit 2000 F  =  66.7 s   bei 30 fps
```

| ⌀ Tempo | Frames/Runde | Runden im Limit |
|---|---|---|
| 0.20 m/s | 1148 | 1.74 |
| 0.60 m/s | 383 | 5.22 |
| 1.00 m/s | 230 | 8.71 |
| 1.90 m/s (v_max) | 121 | 16.54 |

Für **eine** Runde genügt ein Schnitt von 0.115 m/s. Vollgas aus dem Stand
erreicht 1.40 m/s nach 119 Frames und legt in 2000 Frames 121 m zurück, also
15.8 Runden.

Das Limit ist damit so großzügig, dass **jeder nicht-crashende Agent es
zwangsläufig erreicht** — was die obige Wirkung von einem Randfall zum
Regelfall macht. Es wurde bewusst nicht geändert: mit korrekter
Timeout-Behandlung ist die Episodenlänge harmlos, und eine Änderung wäre eine
zusätzliche Variable.

*Suchbegriffe:* time limits in reinforcement learning, partial-episode
bootstrapping, time-aware MDP, episodic vs continuing tasks, absorbing state,
Pardo et al. 2018

---

## 13c. Ringpuffer: die Naht zwischen Ältestem und Neuestem

Aufgetreten beim Umbau aus Abschnitt 9 (Einzelbilder statt fertiger Stapel).
Ein lehrreicher Fehler, weil er erst durch eine Optimierung entstand.

### Wodurch die Abhängigkeit entsteht

Der Standard-Buffer legt bei jedem Eintrag den **fertigen Framestapel** ab.
Jeder Eintrag ist damit für sich vollständig; was daneben liegt, ist
gleichgültig. Überschreiben alter Einträge ist deshalb völlig unproblematisch —
es ist die normale Arbeitsweise eines Ringpuffers.

Die Speicherersparnis entsteht dadurch, nur **ein** Bild abzulegen und den
Stapel beim Ziehen aus den Nachbarindizes zusammenzusetzen. Genau damit wird
jeder Eintrag aber **abhängig von seinen Vorgängern** — und diese Abhängigkeit
bricht an der Stelle, an der der Ring umläuft.

```
Ring der Größe 200, 350 Einträge geschrieben:

Platz     ...  147   148   149  │  150   151   152  ...
schrieb   ...  #347  #348  #349 │  #150  #151  #152 ...
                                 ↑ Naht

Platz 150 (ältester Eintrag) greift für seinen Stapel nach links
auf Platz 149 — und erhält Schreibvorgang #349 statt #149.
200 Schritte in der Zukunft, fremde Episode.
```

Ungültig sind genau `n_stack-1` Indizes ab `pos`.

### Warum SB3 hier nicht schützt

SB3 schließt `self.pos` beim Ziehen aus — aber **nur** bei
`optimize_memory_usage=True`. Ohne das Flag landet die Auswahl in
`BaseBuffer.sample` und zieht gleichverteilt über alle Indizes. Wer aus dem
Standardpfad heraus eine Rekonstruktion baut, verliert diesen Schutz, ohne dass
sich etwas an der Oberfläche ändert.

### Größenordnung und Konsequenz

2 von 500 000 Indizes, also 0.0004 % der gezogenen Übergänge. Praktisch
folgenlos — der Punkt ist nicht der Schaden, sondern die **Testlücke**:

> Beide ursprünglichen Tests liefen mit halbleerem Buffer (150 von 200 bzw.
> 600 von 20 000). Der Ring lief nie um, der fehlerhafte Pfad wurde nie
> ausgeführt, und beide Tests meldeten bitgleiche Ergebnisse.

Ein Test eines Ringpuffers, der den Umlauf nicht erzwingt, prüft die Hälfte der
Implementierung nicht. Der Test wurde entsprechend erweitert (350 Übergänge in
Größe 200, 1.75-facher Umlauf) und prüft dreierlei: dass die übrigen Indizes
weiterhin exakt stimmen, dass die ausgeschlossenen tatsächlich falsch *wären*,
und dass `sample()` sie nicht mehr zieht.

*Suchbegriffe:* circular buffer boundary, frame stacking in replay buffers,
lazy frames, Dopamine OutOfGraphReplayBuffer, off-by-one in experience replay

---

## 13d. Diagnose aus den Tensorboard-Logs: zwei verschiedene Fehler

Lange wurde in diesem Projekt von *einem* Problem gesprochen — „jeder SAC-Lauf
bricht nach seinem Höhepunkt ein". Die Diagnosegrößen aus den Tensorboard-Logs
(`train/critic_loss`, `train/ent_coef`, `rollout/ep_len_mean`,
`rollout/ep_rew_mean`) zeigen, dass es **zwei** sind.

### Fehler A: der Critic divergiert (altes Fahrzeugmodell)

Lauf `1432`, um den Höhepunkt herum:

| Step | ep_rew | ep_len | ent_coef | critic_loss |
|---|---|---|---|---|
| 1 500 000 | 372.3 | 575 | 0.1199 | 41.1 |
| 2 000 000 | **520.8** | 398 | 0.0978 | **22.2** |
| 2 124 594 | −362.1 | 275 | 0.1154 | 57.3 |
| 2 249 189 | 131.4 | 352 | 0.1292 | 1353.4 |
| 2 271 899 | — | — | — | **26 670** |

Der Critic-Verlust springt um **Faktor 1200**, und zwar unmittelbar nachdem er
sein Minimum erreicht hat. Das ist kein allmähliches Verlernen, sondern ein
Stabilitätsereignis. Der Agent bricht nicht ein, weil er die Aufgabe vergisst,
sondern weil seine Wertfunktion explodiert.

Bemerkenswert ist die Reihenfolge: erst das Minimum des Verlusts (22.2, der
niedrigste des ganzen Laufs), dann die Explosion. Ein sehr kleiner TD-Fehler
kurz vor dem Zusammenbruch passt zu Überanpassung — der Critic beschreibt die
Daten im Buffer perfekt und bricht zusammen, sobald die Policy ihn verlässt.

### Fehler B: der Agent lernt gar nicht erst (gemessenes Fahrzeugmodell)

Lauf `2210`, über fünf Millionen Schritte:

| Step | ep_rew | ep_len | ent_coef | critic_loss |
|---|---|---|---|---|
| 1 017 500 | −343.6 | 201 | 0.0512 | 39.9 |
| 3 052 500 | −266.3 | 258 | 0.0620 | 383.2 |
| 4 070 000 | −240.0 | 243 | 0.0578 | 66.1 |
| 4 999 678 | −226.2 | 279 | 0.0580 | 51.0 |

Der Trainings-Reward bewegt sich über den gesamten Lauf zwischen −340 und −226.
**Es gibt keinen Höhepunkt.** Die 875.7, die in der Ergebnistabelle als `best`
stehen, sind eine einzelne günstige deterministische Auswertung, keine
erworbene Fähigkeit.

Daraus folgt eine wichtige methodische Korrektur: `evaluations.npz` allein
täuscht. Eine Bestauswertung ohne entsprechende Bewegung in
`rollout/ep_rew_mean` ist ein Ausreißer, kein Können. Beide Kurven gehören
nebeneinander betrachtet.

### Warum das gemessene Modell die Kamera unbrauchbar macht

Die Beobachtung ist 166×100, herunterskaliert von 830×500 — Faktor 5.
Bewegung je Frame **in dem Bild, das der Agent tatsächlich sieht**:

| v [m/s] | px im Original | px in der Beobachtung |
|---|---|---|
| 0.30 | 2.36 | 0.47 |
| 0.60 | 4.72 | 0.94 |
| 1.00 | 7.87 | 1.57 |
| 1.90 (v_max neu) | 14.95 | 2.99 |
| 5.00 (altes Modell) | 39.33 | 7.87 |
| 12.00 (altes Modell) | 94.40 | 18.88 |

```
Schwelle 1 Pixel je Frame:   v = 0.64 m/s
Auto in der Beobachtung:     4.6 x 1.8 px
```

Unter `force_drag_v1` fuhr das Auto mit 5–12 m/s und verschob sich um 8–19
Pixel je Frame — der Dreierstapel zeigte Bewegung unübersehbar. Unter
`measured_v2` sind es **höchstens 3 Pixel**, und unterhalb von 0.64 m/s ist die
Verschiebung **kleiner als ein Pixel**: die drei gestapelten Bilder sind dann
identisch.

Der Agent startet aus dem Stand. Er muss also durch einen Geschwindigkeitsbereich,
in dem er seine eigene Geschwindigkeit **prinzipiell nicht wahrnehmen kann** —
und wird vom Reward genau dafür bestraft (`v_slow = 0.6`, also fast exakt die
Wahrnehmungsschwelle).

Der Lidar-Experte löst dieselbe Aufgabe unter derselben Physik und demselben
Reward in 12 Minuten, weil er die Geschwindigkeit als exakte Zahl bekommt. Das
grenzt die Ursache ein: **nicht Physik, nicht Reward, nicht der Algorithmus —
die Wahrnehmung.**

### Die Physikkorrektur hat nichts kaputtgemacht, sondern etwas aufgedeckt

Wichtig für die Einordnung: `measured_v2` ist richtig und `force_drag_v1` war
falsch. Das alte Modell hat das Wahrnehmungsproblem nur **verdeckt**, indem es
das Fahrzeug zwanzigfach zu schnell machte. Die guten Zahlen aus Phase 1 sind
teilweise ein Artefakt dieses Fehlers.

### Ansatzpunkte

Gegen Fehler B — Wahrnehmung, die vordringliche Baustelle:

1. **Den Framestapel zeitlich spreizen.** Statt *t, t−1, t−2* die Frames
   *t, t−4, t−8*. Bei 0.3 m/s ergibt das 3.8 Pixel Versatz statt 0.47. Kostet
   nichts, ändert nichts am Prinzip „der Agent lernt aus Bildern", und ein
   Buffer, der Stapel ohnehin über Indexversätze zusammensetzt (Abschnitt 9),
   liefert es fast geschenkt. *Suchbegriffe:* frame skip, dilated frame
   stacking, temporal stride
2. **Höhere Auflösung.** Wirkt, aber schwächer: 300×180 senkt die Schwelle nur
   von 0.64 auf 0.35 m/s.
3. **Geschwindigkeit in die Beobachtung.** Wirksam, aber methodisch ein
   anderes Experiment (siehe 8).

Gegen Fehler A — Stabilität, laut Literatur:

4. **Layer Normalization im Critic** — robusteste Einzelmaßnahme gegen
   Überschätzung, synergiert mit Netz-Resets.
5. **Vollständige Netz-Resets** in Intervallen gegen Plastizitätsverlust.
6. **Kleinere Lernrate für den Entropie-Koeffizienten** gegen
   Entropie-Kollaps. In `2210` fiel `ent_coef` auf 0.0296, in `1432` nur auf
   0.098.
7. Nicht empfohlen: Clipped Double-Q zusätzlich — verschlechtert in
   Kombination mit Netzregularisierung.

Quellen: [Overestimation, Overfitting, and Plasticity in Actor-Critic](https://arxiv.org/html/2403.00514v1),
[Dissecting Discrete Soft Actor-Critic](https://arxiv.org/pdf/2509.09838),
[Distribution-aware sampling of replay buffer](https://www.sciencedirect.com/science/article/abs/pii/S0020025526005712),
[Remember and Forget for Experience Replay](https://arxiv.org/pdf/1807.05827)

*Suchbegriffe:* Q-value divergence, deadly triad, primacy bias, plasticity loss,
dormant neurons, layer normalization critic, periodic network resets, entropy
collapse SAC, extrapolation error off-policy

---

## 14. Offene theoretische Fragen aus diesem Projekt

Punkte, die sich lohnen würden, aber nicht geklärt sind:

1. **Warum bricht jeder SAC-Lauf nach seinem Höhepunkt ein?** Kandidaten:
   falsch behandeltes Zeitlimit (siehe 13b — seit 14.08.2026 korrigiert, damit
   im nächsten Lauf prüfbar), Entropie-Koeffizient läuft weg,
   Critic-Überschätzung, zu kleiner Buffer relativ zur Lauflänge,
   Verteilungsdrift.
   *Suchbegriffe:* policy collapse, Q-value overestimation, entropy tuning in
   SAC, primacy bias, plasticity loss, time limits in RL

2. **Warum sind bei zwei Läufen die deterministischen Eval-Episoden nicht
   identisch** (1–2 % statt 100 %)? Vermutung: die Nullstellen-Klemme im neuen
   Fahrzeugmodell verstärkt Gleitkomma-Unterschiede der GPU-Berechnung.
   *Suchbegriffe:* cuDNN nondeterminism, floating point reproducibility

3. **Wie reagiert ein echter Fahrtregler auf gleichzeitiges Gas und Bremsen?**
   Das Modell nimmt Superposition an, der Agent nutzt das in über der Hälfte
   der Frames. Messbar mit einem Video.

4. **Hilft eine Vorbefüllung des Replay-Buffers mit Experten-Übergängen mehr
   als die reine Backbone-Übertragung?**
   *Suchbegriffe:* SACfD, offline-to-online RL, learning from demonstrations

---

## Referenzwerte dieses Projekts

### Strecke und Fahrzeug

| Größe | Wert | Herkunft |
|---|---|---|
| Rundenlänge (Mittellinie) | 7.66 m | 1807 px aus den Spline-Rändern |
| Außenrand / Innenrand | 8.55 / 6.76 m | `outer_raw_spline.npy`, `inner_raw_spline.npy` |
| Maßstab | 236 px/m | `pixels_per_meter` |
| Zeitschritt | 1/30 s | `dt` |
| Episodenlimit | 2000 Frames = 66.7 s | `carrera_2d_env.py:402` |
| v_max | 1.90 m/s | Modellannahme, nicht messbar (13b, 12b) |
| Beschleunigung 0 → 1.40 m/s | 119 Frames (3.97 s) | `measured_v2` |
| Bremsverzögerung | 0.68 m/s² | Videomessung |

### Ergebnisse

Alle unter `measured_v2` und Reward `v3_rundenzeit`:

| | Runden | ⌀ Tempo | Frames | Ende |
|---|---|---|---|---|
| Lidar-Experte (500k Steps, ~12 min) | 11 | 1.22 m/s | 2000 | kein Crash |
| Vision SAC `2210` (5M Steps, 27 h) | 1 | 0.81 m/s | 256 | Außenwand |
| Vision SAC `2331` (gamma 0.999) | 0 | 0.13 m/s | 33 | Außenwand |

Der Lidar-Experte belegt, dass Physik und Reward lösbar sind. Er bekommt seine
Geschwindigkeit allerdings als exakte Zahl — kein Auflösungsproblem, keine
Subpixelbewegung, keine Quantisierung. Der Unterschied zwischen 12 Minuten und
27 Stunden ist der Preis dafür, aus Pixeln zu lernen.
