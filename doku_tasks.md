# Doku Tasks & Offene Punkte: Carrera RL Projektbericht

Dieses Dokument trackt alle offenen Aufgaben, fehlenden Belege, Re-Trainings, Re-Evaluierungen und formuliert konkrete, strukturierte **Fragen an Claude / den Kollegen (Felix)**, der Zugriff auf die vollständigen Modelle, GPU-Logs und weiteren Trainingsläufe hat.  
---  
**Eigene Anmerkungen**
- Die /cite Zitate in der Doku sind keine Funktion sondern werden als "Code" in der Preview angezeigt - statt nur zb Kürzel. Dazu gibt es keine Verlinkung auf das Quellenverzeichnis.
- Das Superwichitge Buch die RL Bibel auch in Gonkee's Videos zitiert muss als Quelle und erklärung für die Grundlagen herhalten!  
- Schritt 2 in 2.4 - Beschreibt Behavior cloning - korrekt? wenn ja - einmal erwähnen: (Beahavior Cloning)  
- Q Werte bei SAC werde nicht erklärt  
- On- und Off-Policy Bedeutung wird nicht erklärt - warum ist das eine Sampleeffizienter als das andere? Wie lernen beide - was bringt es auch dem Replaybuffer Informationen zu verwenden?  
- An Claude: Welche Learnings aus den genannten Quellen in Projektbericht 2.6 haben wir in diesem Projekt noch nicht angewendet - was könnte und davon weiterbingen?


---

## 1. Statusübersicht der Kapitel (Projektbericht)

| Kapitel | Titel | Status | Offene Aufgaben / Fehlstellen |
|---|---|---|---|
| **Deckblatt & Formalia** | Deckblatt, Erklärung, Abstract | 🟡 In Bearbeitung | Matrikelnummern & Abgabedatum eintragen |
| **Kapitel 1** | Einleitung & Motivation | 🟢 Bereit | Finale Abstimmung der Projektziel-Formulierungen |
| **Kapitel 2** | Theoretische Grundlagen | 🟢 Bereit | Vollständige Zitationen und Querverweise prüfen |
| **Kapitel 3** | Methodik & 2D-Simulationsumgebung | 🟢 Bereit | Diagramm Streckengeometrie / HSV-Masken einbinden |
| **Kapitel 4** | Der 4-Phasen-Trainingsansatz | 🟡 In Bearbeitung | Genaue Datensatzgröße & Pretraining-Loss-Plot ergänzen |
| **Kapitel 5** | Experimentelle Ergebnisse (Zeitleiste) | 🟡 In Bearbeitung | Fehlende Kurven aus `evaluations.npz` & TensorBoard extrahieren |
| **Kapitel 6** | Tiefgehende RL-Erkenntnisse | 🟢 Bereit | Subpixel-Herleitung und Buffer-Benchmarks dokumentiert |
| **Kapitel 7** | Übertragung in die 3D-Simulation (Isaac Sim) | 🔴 Platzhalter | Setup, Szenenaufbau und erste 3D-Ergebnisse fehlen |
| **Kapitel 8** | Fazit, Learnings & Ausblick | 🟢 Bereit | Auf Basis der 2D-Ergebnisse formuliert; 3D-Fazit ergänzen |
| **Verzeichnisse** | Quellen-, Abbildungs-, Tabellenverzeichnis | 🟢 Bereit | Automatische Nummerierung und Abgleich |

---

## 2. Fehlende Belege & Nachträgliche Experimente (Re-Trainings / Re-Evals)

Folgende empirische Belege und Tests müssen noch durchgeführt bzw. aus vorhandenen Logs visualisiert werden:

- [ ] **Task 2.1: Pretraining-Loss-Kurve des CNNs**  
  *Ziel:* Grafischer Nachweis, wie der MSE-Loss über 20 Epochen auf dem Demonstrationsdatensatz fällt und ab wann Überanpassung (Overfitting) einsetzt.
- [ ] **Task 2.2: Vergleichsplots der TensorBoard-Kurven (Critic-Loss & Entropy)**  
  *Ziel:* Gegenüberstellung von Lauf `1432` (Critic-Divergenz) vs. Lauf `1748` (stabiler Lidar-SAC) vs. Lauf `2210` (Vision-SAC).
- [ ] **Task 2.3: Auswertung des gespreizten Framestapels ($t, t-4, t-8$)**  
  *Ziel:* Experimenteller Beweis, dass die Absenkung der Wahrnehmungsschwelle von $0.64\text{ m/s}$ auf $0.16\text{ m/s}$ das Anfahren und Kurvenfahren des Vision-Agenten unter `measured_v2` ermöglicht.
- [ ] **Task 2.4: PPO-Abschlusslauf mit `lr = 5e-5` + Backbone**  
  *Ziel:* Schließen der letzten offenen Vergleichszelle im PPO-Versuchsplan (Vergleich zu SAC bei gleichem Zeitbudget).
- [ ] **Task 2.5: Erzeugung hochauflösender Abbildungen für den Bericht**  
  - [ ] Abbildung Streckenmasken (OpenCV Konturen, Splines, Start/Ziel).
  - [ ] Abbildung Fahrzeugperspektiven: Globale Ansicht ($166\times 100$, 4 Graustufen) vs. Ego-Crop vs. Telemetrieleiste.
  - [ ] Abbildung Trajektorienvergleich: Ideallinie des Lidar-Experten vs. verrauschte Rollouts vs. Schüler-Trajektorie.

---

## 3. Strukturierte Fragen an Claude / Kollegen mit Zugang zu weiteren Modellen & Logs

> **Kontext:** Auf dem Zweitrechner befinden sich historische Checkpoints, ungefilterte TensorBoard-Logs und ggf. 3D-Isaac-Sim-Szenen, die wegen Dateigrößenbeschränkungen nicht vollständig im aktuellen Git-Stand liegen.

### Block A: Datensatz & Behavioral Cloning Pretraining
1. **Genaue Datenpunkte:** Wie viele Frames/Übergänge umfasst der finale Demonstrationsdatensatz für das Pretraining (z.B. Datensatz `20260814_193640`) genau?
2. **Noise Injection Parameter:** Welcher Rauschwert $\sigma$ (Standardabweichung auf Lenkung und Gas) wurde bei den Aufzeichnungsfahrten des Lidar-Experten verwendet?
3. **Pretraining Performance:** Welcher finale Validierungs-MSE wurde auf den 3 Aktionen (Lenkung, Gas, Bremse) erreicht, und wurden die Convolutional-Layer während des anschließenden SAC-Trainings eingefroren (frozen backbone) oder end-to-end mittrainiert?

### Block B: TensorBoard-Verläufe & Detailwerte historischer Läufe
4. **PPO-Läufe (`1601`, `2233`, `2328`):** Liegen die genauen Kurven für `rollout/ep_rew_mean` und `train/explained_variance` für den 90M-Schritte-Lauf `2328` vor?
5. **Kamerarezepte (`2147`, `2210`, `2331`):** Gibt es Aufzeichnungen darüber, ob bei `2210` (5M Steps) zwischenzeitlich Teiltrajektorien von mehr als einer halben Runde gefahren wurden, oder ob das Fahrzeug an denselben Kurvenscheitelpunkten verunfallte?
6. **Lauf `1432` Checkpoint:** Ist der Checkpoint bei genau Schritt 2.0M (Höhepunkt mit 3828 Punkten) separat gesichert, um ihn für Videoaufnahmen/Abbildungen rendern zu können?

### Block C: Übertragung in die 3D-Simulation (NVIDIA Isaac Sim / Isaac Lab)
7. **Stand des 3D-Setups:** Existiert bereits ein USD-Asset der Carrera-Hybrid-Strecke und des Fahrzeugs in Isaac Sim?
8. **Physik-Parameter in 3D:** Welche PhysX-Reibwerte ($\mu_{\text{static}}, \mu_{\text{dynamic}}$) und Reifenmodelle wurden für die 3D-Simulation hinterlegt?
9. **Kameramodell:** Wird in Isaac Sim eine Deckenkamera (Top-Down Orthografisch/Perspektivisch) oder eine fahrzeugfeste Onboard-Kamera simuliert, und welche Render-Pipeline (RTX Real-Time vs. Tiled Rendering) wird für die Observation-Frames verwendet?
10. **Erste Trainingsergebnisse in 3D:** Wurde der Lidar-Experte oder das vortrainierte Vision-CNN bereits in Isaac Lab getestet, und welche Konvergenzzeiten zeigen sich im Vergleich zur 2D-Python-Umgebung?

---

## 4. Checkliste für die Finalisierung des Projektberichts (20–40 Seiten)

- [ ] **1. Deckblatt & Formalia:** Namen, Matrikelnummern, Betreuer, Hochschule Karlsruhe Layout.
- [ ] **2. Abstract & Kurzfassung:** Prägnante Zusammenfassung der Leitidee (Teacher-Student Distillation) und der Subpixel-Erkenntnis.
- [ ] **3. Formeln & Mathematische Notation:** Durchgängige Symbole für MDP, POMDP, Bellman-Gleichungen, SAC-Entropie und Reward-Funktionen.
- [ ] **4. Querverweise & Zitationen:** Alle Paper aus `Quellen und Paper` sauber mit BibTeX-Schlüsseln/Nummern im Text referenziert.
- [ ] **5. Abbildungen & Tabellen:** Platzhalter durch gerenderte PNG-Grafiken und formatierte Tabellen ersetzen.
- [ ] **6. PDF-Export:** Kompilierung via `md_to_pdf.py` mit HKA-Kopfzeile, Fußzeile, Seitennummerierung und klickbaren Inhaltslinks testen.
