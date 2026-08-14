# Offene Punkte

Stand 01.08.2026, nach der Umstellung auf das gemessene Fahrzeugmodell
(`measured_v2`) und Reward `v3_rundenzeit`.

Die Zahlen in Klammern sind gemessen, nicht geschätzt — die zugehörigen
Auswertungen liegen in `Other tools/Auto Parameter messen/Code/`.

---

## 1. Die eigentlichen Blocker

### 1.1 Auflösung der Beobachtung

Das Auto ist in der Beobachtung (166 × 100) nur **4.6 × 1.8 Pixel** groß.
Unterhalb von **0.64 m/s** bewegt es sich weniger als einen Pixel pro Frame:

| v (m/s) | Bewegung pro Frame |
|---|---|
| 0.10 | 0.16 px |
| 0.30 | 0.47 px |
| 0.50 | 0.79 px |
| 0.80 | 1.26 px |

Der Agent kann seine Geschwindigkeit dort also **nicht** aus dem Framestapel
ableiten — genau in dem Bereich, in dem er 33 % seiner Zeit verbringt und in
dem die gesamte Anfahrphase liegt.

Mögliche Wege, kombinierbar:

* Geschwindigkeit direkt in die Beobachtung geben (billigste Änderung).
  `obs_type="multi"` existiert, nutzt aber 84×84-Crop statt 100×166-Global —
  das vorhandene Backbone passt dann nicht mehr. Sauberer wäre eine Variante
  mit globalem Bild **plus** `[v/max_speed, current_steer]`.
* Auflösung erhöhen, etwa auf 250×150 (Auto dann ~7×3 px). Die GPU war nicht
  der Engpass, das wäre bezahlbar.
* Fahrzeugfeste Ansicht (`camera_view="crop"`): Auto wäre 13×5 px. Preis:
  keine weit vorausliegenden Kurven mehr sichtbar.
* Mehr Graustufen (aktuell 4).

### 1.2 Die Policy ist spröde

Zwischen den beiden Renderpfaden unterscheiden sich **0.05 % der Pixel**
(Kantenglättung des Auto-Sprites, cv2 glättet, pygame nicht). Das reicht,
um die Aktionen vollständig zu ändern:

```
hidden-Bild : Gas 0.488  Bremse 0.974  Lenk +0.302
human-Bild  : Gas 0.875  Bremse 0.908  Lenk -0.801
```

Erklärbar, weil die acht abweichenden Pixel ungefähr dem gesamten Fahrzeug
entsprechen (siehe 1.1). Folge: ein Modell, das an der Kantenglättung des
eigenen Renderers scheitert, überträgt nicht auf das echte Auto.

Gegenmittel wäre Augmentierung der Beobachtung beim Training — leichtes
Rauschen, minimale Verschiebungen, Helligkeitsschwankungen. Fehlt derzeit
komplett.

### 1.3 gamma

`gamma = 0.999` hat den Lauf `20260731_2331` zerlegt: Zusammenbruch nach
250.000 Steps, danach 1.25 Millionen Steps flache Linie bei Reward −355 und
Episodenlänge 35.

Ursache vermutlich die Wertskala: bei +2.9 Reward pro Frame wächst die zu
lernende Wertfunktion von ~290 (gamma 0.99) auf ~2900 (gamma 0.999), bei
unveränderter Lernrate 5e-5.

Nächster Versuch: **0.995** (Horizont 6.7 s). Das ist immer noch doppelt so
viel wie die 3.2 s, die das Auto aus 1.3 m/s zum Stehen braucht.

---

## 2. Ungeklärte Modellannahme: Gas und Bremse gleichzeitig

Der Agent betätigt in **über der Hälfte der Frames beides** (gemessen: Gas
0.78, Bremse 0.37 im Mittel). Im Modell wirken sie additiv:

```
dv/dt = gas*A_drive(v) - brake*A_BRAKE - Widerstand
```

Das ist eine **Annahme**, keine Messung — alle Videos zeigten nur entweder
Vollgas oder Vollbremsung. Ein echter Fahrtregler könnte stattdessen die
Bremse gewinnen lassen (Motor kurzschließen) oder den zuletzt empfangenen
Befehl nehmen.

Der PS4-Controller kann beides gleichzeitig senden (zwei Analog-Trigger),
die Eingabe ist also möglich. Offen ist die Reaktion des Autos.

**Zu tun:** zehn Sekunden Video mit beiden Triggern gedrückt, Auswertung mit
`analyse_v10.py`. Vergleich mit dem reinen Vollgas-Video sagt direkt, welches
der drei Verhalten vorliegt.

Solange das offen ist, kämpft der Agent möglicherweise gegen eine Erfindung
des Modells.

---

## 3. PPO-Vergleich sauber wiederholen

Branch `ppo-vergleich`. Die drei vorhandenen PPO-Läufe stammen vom 26.07.
und liefen unter `force_drag_v1` ohne Reward-Version — ein Vergleich mit den
SAC-Läufen unter `measured_v2` ist damit nicht möglich. Das ist der
eigentliche Grund für eine Wiederholung.

```
SAC 1432  : lr 5e-5,  bsp True,   1 Env,   2.5M Steps, 18h49   best  3828.2
PPO 1601  : lr 3e-4,  bsp True,  12 Envs,  2.4M Steps           best  -338.8
PPO 2233  : lr 3e-4,  bsp False, 12 Envs,  2.5M Steps           best  -235.7
PPO 2328  : lr 1e-4,  bsp False, 12 Envs,   90M Steps, 12h09    best    -3.9
```

Zum Aufbau: `bsp` (Backbone-Transplantation) und die Step-Zahl waren
**bewusst gewählt**, keine übersehenen Unterschiede. Ohne BSP schnitt PPO in
den Vorläufen besser ab, und 90M Steps wurden angesetzt, um ungefähr
dieselbe Rechenzeit wie SAC mit 2.5M zu erreichen. Der Vergleich bei
gleichem Zeitbudget ist der sinnvollere Maßstab, weil ein Schritt bei beiden
Verfahren unterschiedlich teuer ist.

Zwei Einschränkungen bleiben:

* **Die Zeiten waren nicht gleich.** 12h09 gegen 18h49, PPO bekam rund ein
  Drittel weniger. Für gleiche Rechenzeit wären eher 130–140M Steps nötig.
* **Die BSP-Aussage stützt sich auf einen Lauf mit vier Auswertungen.**
  1601 (mit BSP) gegen 2233 (ohne) unterscheiden sich um 100 Reward-Punkte,
  beide deutlich negativ. Für eine belastbare Aussage müsste ein BSP-Lauf
  mit vernünftiger Auswertungsdichte dazu — kurz, keine 12 Stunden.

**Vor dem Lauf herzurichten:**

* **`eval_freq` durch `n_envs` teilen.** SB3 zählt `eval_freq` pro Env, aus
  `50000` bei 12 Envs wird also eine Messung alle 600.000 Schritte — 60-mal
  seltener als bei SAC mit einem Env. Weil `best_model` nur bei einer
  Auswertung geschrieben wird, ist dessen Auswahl bei allen bisherigen
  PPO-Läufen systematisch gröber. Das ist der eine klare methodische Fehler
  im bisherigen Vergleich.
* `TOTAL_TIMESTEPS` als eine Variable statt zweier Zahlen (wie in der
  SAC-Zelle bereits umgesetzt)
* `GAMMA` explizit setzen und loggen — steht bei PPO gar nicht drin
* Backbone-Automatik statt des festen Juni-Pfads in `10_Vision_PPO_Global`
* `train_config`-Dict ergänzen, fehlt in einer der beiden PPO-Zellen komplett
* Laufzeit protokollieren, nicht nur die Step-Zahl — sonst lässt sich der
  Vergleich bei gleichem Zeitbudget nachträglich nicht belegen

---

## 4. Kleinere offene Punkte

### 4.1 `bsp` klären

Wird in jeder `train_config.json` geloggt, ist aber **nirgends im Code
auffindbar**. Entweder er spielt eine Rolle und gehört beim Vergleich
kontrolliert, oder er ist ein Überbleibsel und gehört entfernt — wie
`best_lap_bonus` und `slow_lap_penalty`, die ebenfalls geloggt, aber nie
gelesen wurden.

### 4.2 `n_envs` bei Lauf `20260612_2347`

Die Config sagt `1`, das Modellarchiv von SB3 nennt `3`. Belege: der
Eval-Abstand beträgt 30.000 statt 10.000 Steps (SB3 zählt `eval_freq` pro
Env), und der Lauf war bei gleicher Step-Zahl dreimal schneller.

Folge: nur 249.166 Gradientenschritte statt 607.499 bei den übrigen
SAC-Läufen, weil `gradient_steps=1` nicht mit der Env-Zahl mitwuchs. Der
Lauf ist als Vergleich nicht brauchbar.

Noch nicht korrigiert, um nicht stillschweigend Verlaufsdaten zu ändern.

### 4.3 Reward-Boden greift später als früher

`episode_reward <= -3000` beendet die Episode. Mit der alten Klippe
(−15 unterhalb 0.1 m/s) passierte das bei Stillstand nach **6.7 Sekunden**,
mit dem jetzigen Gefälle (max −4) erst nach **25 Sekunden**.

Im Kriechbereich ist das Gefälle strenger als die Klippe (bei 0.2 m/s:
−2.59 gegen +0.08), beim echten Stillstand milder. Falls ein stehendes Auto
zu lange blockiert: `w_slow` von 4.0 auf 15.0, dann greift der Boden wieder
nach 6.7 s, das Gefälle bleibt aber stetig.

### 4.4 Startposition liegt auf der Ziellinie

Start bei (1.91, 0.35) m entspricht (450.8, 82.6) px, die Ziellinie liegt
bei x = 462…468. Die Fahrzeugfront reicht bis x ≈ 461 — gut ein Pixel davor.
Nach jedem Reset gibt es damit einen geschenkten `lap_bonus`.

Bewusst so belassen, damit der Agent den Zusammenhang Ziellinie/Belohnung
früh mitbekommt.

### 4.5 Seed

Im Notebook wird nirgends einer gesetzt (weder `set_random_seed` noch
`seed=`). Die Env nimmt seit der Umstellung einen entgegen und protokolliert
ihn; damit er wirkt, müsste das Notebook ihn an die Env **und** an den
SB3-Konstruktor übergeben. Ohne ihn sind Läufe konfigurierbar, aber nicht
bit-genau wiederholbar.

### 4.6 Reward-Formel nicht vollständig protokolliert

`to_dict()` schreibt die Gewichte und seit `v3_rundenzeit` eine
Versionskennung. Wer die Formel in `calculate()` umbaut, ohne die Gewichte
zu ändern und ohne die Version hochzuzählen, erzeugt jedoch weiterhin ein
identisch aussehendes Log. Ein Hash der Datei wäre die proportionale Lösung
(ein Feld, kein Protokoll pro Frame).

---

## 5. Was das Vortraining tatsächlich überträgt

Vom CNN-Vortraining werden **nur die drei Faltungsschichten** übernommen:

```python
sb3_cnn[0].weight.copy_(pretrained_state_dict['cnn.0.weight'])
sb3_cnn[2].weight.copy_(pretrained_state_dict['cnn.2.weight'])
sb3_cnn[4].weight.copy_(pretrained_state_dict['cnn.4.weight'])
```

Die beiden Linear-Schichten — also der Teil, der aus Bildmerkmalen Gas,
Bremse und Lenkung macht — werden verworfen, und die kopierten Gewichte
werden im Feintuning **nicht eingefroren**.

Übertragen wird also nur die Sehfähigkeit, nicht das Fahrwissen. Von den
11 Runden des Lidar-Experten kommt beim Vision-Agenten nichts an.

Denkbare Steigerungen, aufsteigend nach Wirkung:

1. Faltungsschichten anfangs einfrieren
2. auch die Linear-Schichten übernehmen und den Actor damit initialisieren
3. Replay-Buffer mit Experten-Übergängen vorfüllen (SAC lernt off-policy).
   Dafür müsste der Experte neu aufgezeichnet werden, diesmal mit
   vollständigen Übergängen (`obs, action, reward, next_obs, done`) statt
   nur Bild und Aktion.

---

## 6. Referenzwerte

Zum Einordnen künftiger Läufe, alle unter `measured_v2` und
`v3_rundenzeit` gemessen:

| | Runden | ⌀ Tempo | Frames | Ende |
|---|---|---|---|---|
| Lidar-Experte (500k Steps) | 11 | 1.22 m/s | 2000 | kein Crash |
| Vision SAC `2210` (5M) | 1 | 0.81 m/s | 256 | Außenwand |
| Vision SAC `2331` (1.5M, gamma 0.999) | 0 | 0.13 m/s | 33 | Außenwand |

Der Lidar-Experte zeigt, dass Physik und Reward lösbar sind — er bekommt
seine Geschwindigkeit allerdings als exakte Zahl, ohne Auflösungsproblem.

Messwerkzeug: `python -m src.eval.rauchtest --model <pfad>`
