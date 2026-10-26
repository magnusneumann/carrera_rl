# Spickzettel Präsentation – Carrera RL (Phase 1, 2D)

Quellen: `src/envs/carrera_2d_env.py`, `Notebook_prototyping.ipynb`, `run_trainings.py`, Projektbericht `astra/chapters/02–04`.

---

## 1. PPO und SAC – on-policy vs. off-policy

**Grundbegriffe (ein Satz pro Begriff)**
- **Policy π**: das neuronale Netz, das aus der Beobachtung die Aktion macht (Gas, Bremse, Lenken).
- **Actor-Critic**: ein Netz *handelt* (Actor), ein zweites *bewertet*, wie gut ein Zustand/eine Aktion ist (Critic). PPO und SAC sind beide Actor-Critic und beide **modellfrei** (sie lernen kein Weltmodell, nur aus erlebten Übergängen).

**On-policy vs. off-policy – die Kernfrage: Mit welchen Daten darf ich lernen?**

| | **On-policy (PPO)** | **Off-policy (SAC)** |
|---|---|---|
| Lernt aus | nur Erfahrung der *aktuellen* Policy | auch aus alter / fremder Erfahrung |
| Gedächtnis | keins – Daten werden nach dem Update **verworfen** | **Replay-Buffer** (bei uns 500 000 Übergänge) |
| Folge | robust, aber braucht sehr viele Samples | sample-effizient, kann aber instabil werden |

**Bild für den Vortrag:** PPO ist wie ein Schüler, der nach jeder Übungsstunde sein Heft wegwirft und nur aus der letzten Stunde lernt. SAC hebt alle Hefte auf und wiederholt alte Aufgaben immer wieder.

**PPO (Proximal Policy Optimization)**: sammelt einen Block Erfahrung, macht darauf mehrere Gradientenschritte, „clip“ begrenzt, wie stark sich die Policy pro Update ändern darf → stabil. Danach wird der Block weggeworfen.

**SAC (Soft Actor-Critic)**: maximiert Belohnung **plus Entropie** (bleibt so zufällig wie möglich, solange es gut läuft → exploriert besser). Zwei Critics, das Minimum wird genommen (gegen Überschätzung). Jeder Übergang im Buffer wird vielfach zum Lernen genutzt.

**Unsere Zahlen (Bericht, Abschnitt PPO gegen SAC):**
- Pro 6 144 gesammelten Frames: **SAC ≈ 3 072 Gradientenschritte, PPO = 240** → Faktor ≈ 13.
- Bester PPO-Vision-Lauf: **90 Mio. Schritte, 12,2 h → −3,9 Punkte**.
  SAC-Lauf `2139`: **1,5 Mio. Schritte, 11,2 h → 2 427,5 Punkte**. Gleiches Zeitbudget, SAC klar besser.
- Schwäche SAC: Critic kann auf Bilddaten explodieren (Lauf `1432`: Critic-Loss 22 → 26 670, Leistung bricht ein). Beim Lidar-SAC passiert das nicht → Problem sind mehrdeutige Bilder, nicht SAC an sich.
- Fairness-Hinweis: PPO-Läufe liefen noch unter alter Physik, und `eval_freq` zählt in SB3 *pro Umgebung* → PPO wurde 60× seltener ausgewertet. Der PPO-Bestwert ist eher eine untere Schranke.

---

## 2. Action Space und Observation Space

### Action Space (für alle Varianten gleich)
`Box(3)`, kontinuierlich:

| Index | Bedeutung | Bereich |
|---|---|---|
| 0 | Gas | 0 … 1 |
| 1 | Bremse | 0 … 1 |
| 2 | Lenken | −1 … +1 (→ max. ±30° Lenkwinkel) |

Dazu kommt die **Lenk-Trägheit**: Der Lenkwert darf sich pro Frame um höchstens 0,5 ändern (`max_steer_change`). Das verhindert binäres Hin-und-her-Zucken, das ein reales Servo nicht könnte.

### Observation Space (das, was der Agent „sieht“)

| `obs_type` | `camera_view` | Form | Inhalt |
|---|---|---|---|
| `lidar` | – | `Box(4)` float | `[v/v_max, d_links, d_mitte, d_rechts]`: 3 Raycasts bei −25°/0°/+25° |
| `vision` | `crop` | `1×84×84` uint8 | fahrzeugfester Ausschnitt (150×150 px → 84×84), Auto immer mittig, nach oben |
| `vision` | `global` | `1×100×166` uint8, später `1×150×250` | ganze Strecke von oben, Auto ist ein kleiner Sprite |
| `multi` | (immer crop) | Dict: `image 1×84×84` + `proprioception [v/v_max, Lenkwert]` | Hybrid: Bild + 2 Zahlen |

- **Frame-Stacking** (`VecFrameStack`, n_stack=3) macht aus 1 Kanal 3 Kanäle: die letzten 3 Bilder übereinander. Das Netz soll aus der Verschiebung zwischen den Bildern Tempo und Richtung ableiten.
- **POMDP-Argument:** Ein Einzelbild zeigt *wo* das Auto ist, aber nicht *wie schnell* und *wohin* es fährt. Die Umgebung ist für den Agenten nur teilweise beobachtbar. Stacking oder Zusatzparameter sollen das zurückholen.
- Größenordnung: Lidar-Policy ≈ 10 000 Parameter, Vision-Policy bei 166×100 ≈ **16,3 Mio.**, bei 250×150 ≈ **41 Mio.**

---

## 3. Steps / Frames beim Training erklären

**Kernsatz:** *1 Step = 1 Frame = 1/30 Sekunde simulierte Zeit.*

- Bei jedem `env.step(action)` wird die Physik um `dt = 1/30 s` weitergerechnet, das Bild neu gezeichnet und eine Belohnung vergeben.
- `total_timesteps` zählt diese Steps **über alle Umgebungen zusammen**. 12 Umgebungen × 300 000 Steps = 3,6 Mio. Steps.
- Frame-Stacking ändert daran nichts: Pro Step kommt genau **ein** neues Bild dazu, die zwei älteren werden nur mitgeschoben.
- Simulierte Zeit ≠ Rechenzeit:
  - Lidar-SAC: 500 000 Steps in **54 min**
  - Vision-SAC 166×100: ≈ 135 000 Steps/h, 250×150: ≈ 75 000 Steps/h → Millionen Steps = Stunden bis Tage
- Faustregel für die Folie: **Lidar braucht ~50 000 Steps bis zur vollen Episodenlänge, Vision Millionen** → Bildbeobachtung kostet mindestens Faktor 70.
- Planungshorizont: γ = 0,99 → ≈ 1/(1−γ) = **100 Frames ≈ 3,3 s** Vorausschau.

---

## 4. „Warum nicht mehrere Trainingsräume parallel?“

**Klarstellung zuerst:** Es *gab* parallele Umgebungen, nämlich bei **PPO 12 Umgebungen** (`SubprocVecEnv`, `n_envs = 12` in `run_trainings.py` und den Notebook-Zellen 36, 60, 61). Erst bei den **SAC-Läufen** wurde auf **1 Umgebung** gegangen (`n_envs = 1`, Notebook-Zelle 58; ein Lauf, `2147`, hatte zwischendurch 3).

**Warum PPO viele Umgebungen braucht:** PPO wirft seine Daten nach jedem Update weg. Damit ein Update genug und vielfältige Daten hat, sammelt man gleichzeitig in vielen Umgebungen (12 × 512 = 6 144 Frames pro Runde). Parallelität ist bei PPO der Hebel für Tempo.

**Warum bei SAC eine Umgebung reicht bzw. besser ist:**
1. **Engpass ist das Lernen, nicht das Sammeln.** SAC macht pro Frame einen Gradientenschritt auf einem 512er-Batch (≈ 50 ms). Die Simulation ist viel schneller. Mehr Umgebungen würden nur Daten schneller in den Buffer kippen, als sie verarbeitet werden.
2. **Der Replay-Buffer sorgt schon für Vielfalt.** Gemischt wird über 500 000 gespeicherte Übergänge, nicht über parallele Räume.
3. **Speicher:** Der Buffer liegt im RAM: bei 166×100 schon **15,5 GB** (nach eigener Optimierung, vorher 23,2 GB), bei 250×150 **34,9 GB**. Zusätzliche Prozesse mit eigenem Rendering kosten weiteren Speicher.
4. **Die Kennzahlen bleiben verständlich.** SB3 zählt `eval_freq` und Ähnliches *pro Umgebung*. Mit 12 Umgebungen wurde bei PPO deshalb unbemerkt 60× seltener ausgewertet.

**Für die Folie:** *„PPO skaliert über Breite (viele Umgebungen), SAC über Gedächtnis (Replay-Buffer).“*

Nebenbei: 12 Pygame-Umgebungen haben unter Windows die Fenster-Handles erschöpft → Umstieg auf Headless-Rendering mit OpenCV.

---

## 5. „Warum gibt es keine Episoden?“

**Doch, es gibt Episoden.** Sie sind nur nicht die Einheit, in der das Training gemessen wird.

**Eine Episode endet, wenn** (`step()` in der Env):
- **Crash** an Innen- oder Außenwand → `terminated` (echter Endzustand, Wert = 0)
- Episoden-Belohnung ≤ −3000 → `terminated` (+ Strafe −100)
- **2000 Frames erreicht (≈ 66,7 s)** → `truncated` (Zeitlimit, *kein* echter Endzustand)

Danach setzt das VecEnv automatisch per `reset()` zurück, und das Training läuft einfach weiter.

**Warum man trotzdem in Steps statt Episoden denkt:**
- Episoden sind **unterschiedlich lang**: anfangs crasht der Agent nach 30 Frames, später fährt er 2000. „1000 Episoden“ wäre deshalb keine vergleichbare Menge Erfahrung, „1 Mio. Steps“ schon.
- PPO und SAC updaten nach festen **Step-Zahlen** (`n_steps`, `train_freq`), nicht am Episodenende. Ein Update kann mitten in einer Episode passieren.
- Es gibt kein „Epochen“-Konzept wie beim überwachten Lernen, weil es keinen festen Datensatz gibt. Der Agent erzeugt seine Daten selbst.

**Feinheit, die gut ankommt:** Zeitlimit ≠ Crash. Bei Crash ist das Lernziel `y = r`. Beim Zeitlimit wäre das Auto weitergefahren, also `y = r + γ·Q(s')`. Bei uns stand `handle_timeout_termination` lange auf False (erzwungen durch `optimize_memory_usage=True`), was Zeitlimit wie Crash behandelt. Weil der Agent keine Uhr sieht, verschmiert dieser Fehler über alle „guten“ Zustände. Inzwischen ist das behoben.

---

## 6. Welche Input-Konzepte wurden trainiert?

Zwei Achsen plus Hybrid (Bericht, Abschnitt Eingabekonzepte):
- **Zeitliche Tiefe:** Single-Frame vs. 3er-Stack (später auch gespreizt)
- **Perspektive:** Ego-Crop (fahrzeugfest) vs. global (ganze Strecke)
- **Hybrid:** Bild + Geschwindigkeit + Lenkwinkel

| # | Konzept | Laufnamen (models/, tensorboard_logs/) | Algorithmus |
|---|---|---|---|
| 1 | **Lidar / Raycasts** (Referenz, privilegiert) | `01_Lidar_Baseline`, `12_Lidar_SAC_Kontrolle` | PPO, SAC |
| 2 | Single-Frame, Crop | `02_Vision_SingleFrame`, `04_Vision_Crop_1Frame` | PPO |
| 3 | Single-Frame, global | `02_Vision_SingleFrameGlobal`, `05_Vision_Global_1Frame` | PPO |
| 4 | Hybrid: Crop + v + Lenkung | `03_Hybrid_MultiInput(_Fast)`, `07/08_Hybrid_Multi` | PPO |
| 5 | 3er-Stack, Crop | `04_Vision_FrameStack_3`, `06_Vision_Crop_3Stack` | PPO |
| 6 | 3er-Stack, global 166×100 | `07_Vision_Global_3Stack`, `10/11_Vision_PPO_*`, `09_…` (Juli) | PPO, SAC |
| 7 | 3er-Stack, global **250×150** | `09_Vision_SAC_Finetuned_…2230/2254/2215` | SAC + Vortraining |
| 8 | 3er-Stack, global, **gespreizt** (t, t−4, t−8) | `09_…2307` | SAC + Vortraining |

**Aktuelle Pipeline (das, was funktioniert):**
Lidar-Lehrer (PPO) → fährt mit Rauschen und zeichnet Bildstapel + Korrekturaktion auf → CNN-Vortraining (Behavioral Cloning, nur die Conv-Schichten werden übernommen) → **SAC-Feintuning auf global 3-Stack 250×150**.

---

## 7. Warum sind fast alle Input-Konzepte gescheitert?

**Kurzantwort für die Folie:** *Nicht der Algorithmus war das Problem, sondern die Wahrnehmung. Der Agent konnte seine eigene Geschwindigkeit nicht sehen.*

### Die Gründe pro Konzept

**Single-Frame (Crop und global): prinzipiell unlösbar (POMDP).**
Ein Einzelbild enthält weder Tempo noch Fahrtrichtung. Für dasselbe Bild gibt es je nach Tempo verschiedene richtige Aktionen, also keine eindeutige Abbildung Bild → Aktion. Konvergierte kaum.

**Ego-Crop: Sichtweite kürzer als Bremsweg.**
Nach der Umstellung auf gemessene Physik (`measured_v2`) zeigt der Ausschnitt nur 0,32 m nach vorn. Der Bremsweg beträgt aus 0,5 m/s 0,38 m, aus 1,2 m/s 1,52 m. Der Agent sieht die Wand erst, wenn Bremsen nicht mehr reicht. Mit dem alten, viel zu stark motorisierten Auto (Bremsweg 6 cm) ging das noch, deshalb sah Crop im Juni gut aus.

**Global 3-Stack 166×100: Subpixel-Problem.** *(Das ist der Hauptbefund)*
- 1 Pixel in der Beobachtung = 21,2 mm. Das Auto ist nur **4,6 × 1,8 px** groß.
- Damit es sich zwischen zwei Frames um ≥ 1 px bewegt, braucht es **v ≥ 0,64 m/s**.
- Beim Anfahren (≈ 84 Frames bis 0,6 m/s) sind die 3 gestapelten Bilder **identisch**. Der Agent ist blind für die eigene Bewegung und wird gleichzeitig für Langsamfahren bestraft (v_slow = 0,6 m/s liegt genau auf der Schwelle).
- Das alte Fahrzeugmodell hatte das *verdeckt*: bei 5–12 m/s bewegte sich das Auto 8–19 px pro Frame.

**Beweis per Kontrollversuch (`12_Lidar_SAC_Kontrolle`):** Gleiche Physik, Belohnung, SAC, Buffer, γ, nur die Beobachtung wurde gegen Lidar-Zahlen getauscht → **11 Runden nach 54 min**. Der Kamera-Agent hatte nach 27 h 0 Runden. Damit sind alle anderen Ursachen ausgeschlossen.

**Ablation:** Dem fertigen Lidar-Agenten nur die Geschwindigkeit verfälschen (null, eingefroren, verrauscht, auf 4 Stufen gerastert) → jedes Mal **0 Runden**. Verrauschte Abstände bei exakter Geschwindigkeit → noch 2,7 Runden. **Die Geschwindigkeit trägt die Leistung**, und die Kamera-Agenten landen genau im Bereich „Lidar mit kaputtem Tempo-Signal“.

**PPO auf Bildern: zu sample-ineffizient** (siehe Abschnitt 1).

**Gespreizter Stack (t, t−4, t−8): negatives Ergebnis.** In der Theorie sinkt die Schwelle auf 0,16 m/s, in der Praxis lag der Lauf in 150 Auswertungen nie positiv. Mögliche Gründe (ungeprüft): Die Netze erkennen den „Schmierstreifen“ bei Abstand 1 leichter als drei getrennte Flecken, und das älteste Bild (0,27 s) ist schon veraltet.

**Was dann funktioniert hat:** **Auflösung 250×150** (Schwelle 0,42 m/s, Auto ≈ 7×3 px) + SAC + Vortraining + **streuende Startpunkte**.
- `2215` (streuender Start): **3,1 Runden aus zufälligen Startlagen**, das erste ehrliche Ergebnis.
- `2254` sieht mit 9 Runden besser aus, wurde aber mit festem Start trainiert und ausgewertet. Aus zufälligen Startlagen schafft er nur 0,6 Runden, er hat eine Bahn auswendig gelernt.

### Weitere Stolpersteine (falls Zeit)
- **Henne-Ei-Problem:** Ein Netz mit 16 Mio. zufälligen Gewichten crasht nur, lernt deshalb nicht, was im Bild wichtig ist, und lernt deshalb nicht fahren → Vortraining aus Demonstrationen hilft.
- **γ = 0,999** ließ das Training kollabieren (Wertskala ×10, Lernrate unverändert).
- **Renderpfad:** OpenCV und Pygame unterscheiden sich in 0,05 % der Pixel, also etwa so viele Pixel, wie das ganze Auto belegt. Das reichte, damit der Agent komplett andere Aktionen ausgab.
- **Rundenzeit-Bonus** hat vier Wochen lang nie gefeuert (Bug).

---

## 8. Wurde „global + 1 Frame + Parameter“ trainiert?

**Nein, das lässt sich im Code eindeutig nachvollziehen.**

- **Global + 1 Frame** wurde trainiert, aber **ohne Parameter**: `05_Vision_Global_1Frame` (`obs_type="vision"`, `camera_view="global"`, `run_trainings.py`) und `02_Vision_SingleFrameGlobal` (Notebook-Zelle 32).
- **Bild + Parameter** (Hybrid, `obs_type="multi"`) gab es **nur mit Ego-Crop**:
  - Alle Hybrid-Läufe (`03_Hybrid_MultiInput`, `03_…_Fast`, `07/08_Hybrid_Multi`) rufen die Env ohne `camera_view` auf, also mit dem Default `"crop"`, oder ausdrücklich mit `"crop"` (`08_Hybrid_Multi`).
  - Im Env ist der Bildanteil des Hybrid-Beobachtungsraums fest auf **`1×84×84`** gesetzt (`carrera_2d_env.py`, `obs_type == "multi"`), und zwar in **allen** Versionen der Datei in der Git-Historie.
  - Ein Aufruf `obs_type="multi", camera_view="global"` würde gar nicht funktionieren: Die Kamera liefert dann 100×166, der Beobachtungsraum erwartet 84×84.

**Warum das für die Präsentation interessant ist:** Genau diese Kombination ist nach den eigenen Befunden der naheliegendste nächste Versuch. Die Ablation zeigt, dass dem Kamera-Agenten vor allem die **Geschwindigkeit** fehlt, und der Hybrid-Input liefert sie exakt, für 8 Byte statt mehrerer Kilobyte (ω = v·tan δ / L: wer v und δ kennt, hat die volle Bewegungsinformation). Global + 1 Frame + `[v, Lenkwert]` hätte die globale Übersicht *und* das Tempo-Signal und würde das Subpixel-Problem komplett umgehen. Für den späteren Realbetrieb müsste v dann aus der Deckenkamera per CV geschätzt oder vom Fahrzeug gemeldet werden.
→ Guter Punkt für „Ausblick / offene Experimente“.

---

## Drei Sätze zum Mitnehmen
1. **Die Aufgabe ist lösbar.** Mit exakten Zahlen (Lidar) fährt der Agent nach Minuten 11 fehlerfreie Runden.
2. **Der Engpass ist die Wahrnehmung.** Bei 166×100 Pixeln kann der Agent seine eigene Geschwindigkeit nicht sehen, und das lässt sich *vor* dem Training ausrechnen.
3. **SAC + Vortraining + höhere Auflösung + streuende Startpunkte** ergibt den ersten ehrlich fahrenden Kamera-Agenten (3,1 Runden aus zufälligen Startlagen).
