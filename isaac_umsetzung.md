# Umsetzung in Isaac Sim / Isaac Lab

Was aus der 2D-Umgebung übernommen werden muss, was neu gebaut wird, und welche
Fallen dort wieder auftreten. Beschrieben ist **nur** der Endstand aus der Zelle
`Für den Rechner ausgelegt - großes Training`.

Stand 21.08.2026.

---

## 1. Die Architektur in Zahlen

```
Beobachtung    3 gestapelte Graustufenbilder, 250 x 150, aufeinanderfolgend
               Vogelperspektive der ganzen Strecke, 4 Graustufen (0/64/128/192)
               uint8, Form (3, 150, 250)

Aktion         [Gas, Bremse, Lenken]   Box([0, 0, -1], [1, 1, 1])
               Lenkwert aendert sich um hoechstens 0.5 je Frame

Algorithmus    SAC, CnnPolicy (NatureCNN)
               41.0 Mio Parameter gesamt, davon Actor 13.5, Critic 13.7

Takt           dt = 1/30 s, Episodenlimit 2000 Frames (66.7 s)
```

Hyperparameter:

```
gamma            0.99          Horizont 100 Frames = 3.3 s
learning_rate    5e-05
batch_size       512
buffer_size      500 000
train_freq       2             ein Gradientenschritt je zwei Umgebungsschritte
gradient_steps   1
learning_starts  5 000
n_envs           1
```

### Fahrzeugmodell (aus Videomessung, nicht geraten)

```
dv/dt = gas * A(v) - brake * a_brake - (r0 + r1 * v)
A(v)  = min(0.30 + 0.506 v,  2.05 - 0.717 v)     Haftung / Motor
a_brake = 0.167     r0 = 0.05     r1 = 0.336
v_max = 1.90 m/s    rueckwaerts gesperrt
Reibwert mu = 0.3   Radstand 58 mm   Lenkwinkel 30 Grad
```

In Isaac ersetzt PhysX diese Gleichung. **Das ist der Punkt, an dem die meisten
Zahlen dieses Projekts ungültig werden** — siehe Abschnitt 4.

### Reward

```
Tempo            + 2 * v^2                     jeder Frame
Langsam-Strafe   - 4 * (0.6 - v) / 0.6         nur unter 0.6 m/s
Lenk-Glaettung   - 0.5 * |Lenkaenderung|       jeder Frame
Rundenbonus      + 50                          je Ueberquerung der Ziellinie
Rundenzeit       + 5 * (Bestzeit - neue Zeit)  nur bei Verbesserung
Crash            - 300 und Episodenende
falsche Richtung - 30, dazu -5*v bei v > 0.1
```

### Startbedingungen

```
Startpunkt streut 0.9 m nach hinten entlang der Geraden vor der Ziellinie
und 0.08 m nach jeder Seite. Richtung und Anfangstempo bleiben fest.
```

---

## 2. Was direkt übernommen werden kann

**Der Algorithmus samt Hyperparametern.** SAC mit diesen Werten ist in dieser
Aufgabe erprobt. `gamma = 0.99` sollte man nicht anheben, ohne die Lernrate
mitzuziehen (siehe `rl_erkenntnisse.md`, Abschnitt 4).

**Die Reward-Formel.** Sie ist lösbar — der Lidar-Agent erreicht damit 11 Runden.
Die Gewichte hängen aber an der Fahrzeugdynamik: `v_slow = 0.6` ist auf ein
Fahrzeug mit v_max 1.9 abgestimmt, und die Crash-Strafe von −300 entspricht rund
150 Frames Fahrertrag. Beides muss nachgerechnet werden, wenn PhysX andere
Geschwindigkeiten liefert.

**Die Aktionsdefinition.** Gas, Bremse und Lenken getrennt, mit begrenzter
Lenkgeschwindigkeit. Das entspricht dem echten Fahrzeug.

**Der Einzelbild-Replay-Buffer** (`src/utils/framestapel_buffer.py`) ist
unabhängig von der Umgebung und spart bei Bildbeobachtungen ein Drittel
Speicher. Bei mehreren parallelen Umgebungen wird das relevanter, nicht weniger.

---

## 3. Was neu gebaut werden muss

### Die Beobachtung

Die Vogelperspektive der ganzen Strecke ist in Isaac eine **Kamera über dem
Aufbau**, kein gerendertes 2D-Bild. Zu klären:

- Kameraposition und Blickwinkel so, dass die ganze Strecke im Bild ist
- Auflösung: aktuell 250×150. In Isaac kostet Rendern deutlich mehr, das wird
  die bestimmende Größe für den Durchsatz.
- Graustufen und Quantisierung: aktuell vier Stufen. Die Quantisierung spart
  **keinen Speicher** (uint8 bleibt uint8) und ist damit optional.
- Der Framestapel muss dieselbe zeitliche Basis haben wie hier: drei
  aufeinanderfolgende Frames bei 30 Hz.

### Die Fahrphysik

PhysX statt der gemessenen Gleichung. Damit ändern sich Beschleunigung,
Bremsweg, Höchstgeschwindigkeit und Kurvengrenztempo — und mit ihnen jede
Einstellung, die auf Zeit oder Geschwindigkeit beruht.

Sinnvoll wäre, das gemessene Modell als **Prüfstein** zu behalten: dieselben
Vollgas- und Bremsversuche in Isaac fahren und mit
`Other tools/Auto Parameter messen/` vergleichen. Weicht Isaac stark ab, ist
entweder die Parametrierung falsch oder die Messung war es.

### Kollision und Rundenzählung

Aktuell zwei Shapely-Polygone für die Ränder und eine Linie für Start/Ziel. In
Isaac sind das Kollisionskörper und ein Trigger-Volumen. Die Rundenlogik selbst
kann übernommen werden, inklusive der Reparatur aus Abschnitt 5.

### Zurücksetzen

Die streuenden Startpunkte müssen mitkommen. In Isaac ist das eher einfacher,
weil sich Posen direkt setzen lassen.

---

## 4. Fallen, die dort wieder auftreten

Alles in diesem Abschnitt ist in diesem Projekt tatsächlich passiert und hat
jeweils Tage gekostet.

### Die Wahrnehmungsgrenze neu ausrechnen

Der Framestapel existiert nur, damit der Agent sein Tempo ablesen kann. Das
funktioniert nur, wenn sich zwischen zwei Bildern etwas messbar verschiebt:

```
Verschiebung je Frame = v * (Pixel je Meter) * dt
```

Bei 250×150 in dieser Strecke sind das erst ab **0.42 m/s** mehr als ein Pixel.
Der Unterschied zwischen „Kurve geht" und „geht nicht" lag bei **0.17 Pixel** —
für den Agenten unsichtbar.

**In Isaac diese Rechnung mit der neuen Kamera und den neuen Geschwindigkeiten
wiederholen, bevor trainiert wird.** Ergibt sie wieder Bruchteile eines Pixels,
ist das Ergebnis vorhersehbar.

Der naheliegende Ausweg — die Bilder zeitlich spreizen — wurde hier getestet und
war **schlechter** (`zeitleiste.md`, Phase 6).

### Feste Startpunkte erzeugen auswendig gelernte Trajektorien

Der größte Einzelbefund. Mit festem Startpunkt erreichte ein Agent 5080 Punkte
und 9 Runden — aus zufälligen Startlagen fiel derselbe Agent auf −1829 und
**0.6 Runden**. Er hatte keine Fahrpolitik gelernt, sondern eine Bahn abgespult.

```
                        fester Start        streuender Start
Agent mit festem Start      5080.5                 -1829.2
Agent mit Streuung          1027.6                  1545.7
```

**Von Anfang an mit streuenden Startpunkten trainieren.** Sonst misst man etwas,
das keine Fahrfähigkeit ist, und merkt es erst spät.

Nebeneffekt: ohne Streuung sind alle Auswertungsepisoden identisch, die
Standardabweichung ist exakt 0, und man hält einen Einzelwert für einen
Mittelwert.

### Auswertung auf demselben Gerät wie das Training

Dasselbe Modell, dieselbe Umgebung, nur CPU statt GPU:

```
CPU     216 Frames,  0 Runden, Reward  -273.7
CUDA   2000 Frames,  9 Runden, Reward  5080.5
```

Gleitkomma-Unterschiede um 1e-7 schaukeln sich über eine Episode zu einem völlig
anderen Verlauf auf. Diese Sprödigkeit ist eine Eigenschaft der Policy, nicht
ein Fehler — sie wird in Isaac nicht verschwinden.

### Zeitlimit ist kein Endzustand

`handle_timeout_termination=True` setzen. Sonst lernt der Critic, dass die Welt
am Episodenlimit aufhört und der Zustand dort wertlos ist. In SB3 erzwingt
`optimize_memory_usage=True` das falsche Verhalten, ohne zu warnen.

### Jede zeitbasierte Einstellung hängt an der Fahrzeugdynamik

Ändert sich die Physik, müssen mitgeprüft werden: Diskontfaktor, Episodenlänge,
Framestapel, Zeitlimits, `v_slow`, Crash-Strafe im Verhältnis zum Fahrertrag.
In diesem Projekt wurde die Physik korrigiert und alles andere blieb stehen —
das hat Wochen gekostet.

### Alles protokollieren, was den Lauf bestimmt

`train_config.json` enthält hier Physik, Reward, Kamera, Startstreuung,
Hyperparameter, Git-Commit, Hardware und Laufzeit. Ohne das lässt sich nach
zwei Wochen nicht mehr sagen, warum zwei Läufe verschiedene Zahlen haben.
Mehrfach war genau das der Grund, warum ein Vergleich wertlos wurde.

---

## 5. Fehler, die hier gefunden wurden und dort nicht wieder eingebaut werden sollten

**Rundenzeit über die erste Zielüberfahrt.** Der Startpunkt lag 14 Pixel vor der
Linie, die erste Überquerung dauerte 9 Frames und wurde als Rundenzeit gewertet.
Danach konnte keine echte Runde (rund 172 Frames) das unterbieten — der
Verbesserungsbonus feuerte vier Wochen lang nie. Die erste Überquerung darf die
Zeitmessung nur **starten**.

**Seed bei jedem Reset neu setzen.** Dann zieht der Zufallsgenerator jedes Mal
dieselbe Zahl, und streuende Startpunkte sind in jeder Episode identisch. Einmal
säen genügt; die Folge bleibt reproduzierbar.

**Zwei Umgebungen für die Datenaufzeichnung.** Wenn ein Fahrer-Env und ein
Kamera-Env parallel laufen, müssen sie synchron bleiben. Zufällige Startpunkte
entkoppeln sie, und Bild und Aktion gehören nicht mehr zusammen.

**Aufzeichnung über einen anderen Renderpfad als das Training.** Hier waren es
Pygame gegen cv2 — 0.04 % der Pixel unterschiedlich, genug für 0.13 Lenkabweichung.

---

## 6. Was hier offen geblieben ist

Diese Fragen sind in 2D **nicht** beantwortet und stellen sich in Isaac erneut:

1. **Wovon der Kamera-Agent wirklich profitiert.** Auflösung 166×100 → 250×150
   brachte am Endergebnis nichts, nur Effizienz. Der große Sprung kam mit dem
   reparierten Rundenzeit-Bonus, aber gebündelt mit einem neuen Backbone.

2. **Ob das Vortraining etwas bringt.** Die übertragenen Faltungsschichten
   messen sich nicht besser als zufällige Anfangsgewichte (`src/eval/backbone_probe.py`,
   R² 0.735 gegen 0.748). Ein Lauf ohne Vortraining wurde nie gefahren.

3. **Warum das vortrainierte Netz nie selbst fahren konnte.** Vier Erklärungen
   wurden geprüft und widerlegt. Übrig bleibt Verteilungsdrift, gegen die nur
   DAgger hilft.

4. **Warum SAC-Läufe nach dem Höhepunkt einbrechen.** Bei `1432` explodierte der
   Critic-Verlust von 22 auf 26 670. Der Lidar-Kontrollversuch zeigte das nicht.

---

## 7. Reihenfolge für den Aufbau

```
1  Fahrzeug in Isaac parametrieren, gegen die Videomessung pruefen
2  Kamera setzen, Wahrnehmungsgrenze ausrechnen (Abschnitt 4)
3  Kollision, Ziellinie, Rundenzaehlung
4  Zuruecksetzen mit streuenden Startpunkten
5  Reward uebernehmen, Gewichte an die neuen Geschwindigkeiten anpassen
6  Lidar-Variante zuerst trainieren - sie ist der Beleg, dass Physik und
   Reward loesbar sind, und kostet in 2D nur 12 Minuten
7  erst dann die Bildvariante
```

Schritt 6 ist der wichtigste Rat aus diesem Projekt. Er hat hier in 45 Minuten
geklärt, was vorher wochenlang unklar war: dass die Aufgabe lösbar ist und das
Problem in der Wahrnehmung sitzt.
