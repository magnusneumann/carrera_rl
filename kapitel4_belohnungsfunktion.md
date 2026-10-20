# 4 Belohnungsfunktion

## 4.1 Zielsetzung

Die Belohnungsfunktion übersetzt das Ziel der Aufgabe in ein skalares Signal,
das der Agent in jedem Zeitschritt erhält. Für die vorliegende Aufgabe lautet
dieses Ziel: möglichst viele Runden in möglichst kurzer Zeit zu fahren, ohne die
Strecke zu verlassen und ohne die Lenkmechanik unnötig zu belasten.

Im Reinforcement Learning existiert dabei kein getrennter Mechanismus für
Belohnung und Bestrafung. Beides sind Werte derselben Skalarfunktion; negative
Werte werden üblicherweise als *penalty* bezeichnet. Die Trennung in den
folgenden Abschnitten ist daher eine Frage der Darstellung, nicht der
Implementierung.

Die Funktion ist als eigene Klasse `RewardCalculator` implementiert und über ein
Versionskennzeichen identifizierbar. Dieses kennzeichnet die *Formel*, nicht nur
die Parameterwerte — ohne diese Unterscheidung lassen sich Trainingsläufe nicht
auseinanderhalten, bei denen die Gewichte gleich blieben, die Berechnung sich
aber änderte. Der hier beschriebene Stand trägt die Kennung `v3_rundenzeit`.

## 4.2 Aufbau der Belohnungsfunktion

Die Funktion setzt sich aus zwei Arten von Bestandteilen zusammen: solchen, die
in jedem Zeitschritt ausgewertet werden (*dense*), und solchen, die nur bei
bestimmten Ereignissen auslösen (*sparse*). Erstere geben dem Agenten in jedem
Frame eine Rückmeldung und ermöglichen so überhaupt erst das Erlernen einer
kontinuierlichen Fahrweise; letztere kodieren das eigentliche Aufgabenziel.

### 4.2.1 Fortlaufende Bewertung

| Bestandteil | Formel | Parameter |
|---|---|---|
| Geschwindigkeit | `+v² · w_speed · 2.0` | `w_speed = 1.0` |
| Langsam-Gefälle | `−w_slow · (v_slow − v) / v_slow`, nur bei `v < v_slow` | `v_slow = 0.6`, `w_slow = 4.0` |
| Lenkglättung | `−|Δ_steer| · w_smooth` | `w_smooth = 0.5` |

Der **Geschwindigkeitsterm** gilt ohne Untergrenze ab Stillstand und wächst
quadratisch. Die quadratische Form ist bewusst gewählt: eine Verdopplung der
Geschwindigkeit vervierfacht die Belohnung, wodurch zügiges Fahren
überproportional gegenüber vorsichtigem Kriechen bevorzugt wird.

Das **Langsam-Gefälle** ergänzt ihn nach unten. Die Strafe wächst linear vom Wert
null bei `v_slow` auf `w_slow` bei Stillstand. Die beiden Parameter haben dabei
verschiedene Rollen: `v_slow` legt die Schwelle fest, unterhalb derer bestraft
wird, `w_slow` die maximale Höhe der Strafe. Zusammen mit dem
Geschwindigkeitsterm ergibt sich ein Nulldurchgang bei etwa 0.52 m/s —
unterhalb dieser Geschwindigkeit ist jeder Zeitschritt netto negativ.

Die **Lenkglättung** bestraft jede Änderung des Lenkwinkels gegenüber dem
vorherigen Frame. Da die Änderung pro Schritt auf `max_steer_change = 0.5`
begrenzt ist, beträgt die maximale Strafe 0.25 pro Frame. Sie dient nicht der
Fahrleistung, sondern dem Bauteilschutz: der Lenkwinkel wird beim realen
Fahrzeug von einem Servo gestellt, dessen mechanische Belastung mit Zahl und
Amplitude der Stellbewegungen steigt.

### 4.2.2 Ereignisbasierte Bewertung

| Bestandteil | Auslöser | Wert |
|---|---|---|
| Crash | Berührung einer Streckenbegrenzung | −300, beendet die Episode |
| Rundenbonus | Vorwärtsüberquerung der Ziellinie | +50 |
| Rundenzeitverbesserung | neue Bestzeit innerhalb der Episode | `+(bisher_beste − aktuell) · 5.0` |

Der **Rundenbonus** kodiert das eigentliche Aufgabenziel. Er wird bei jeder
Vorwärtsüberquerung der Ziellinie vergeben, einschließlich der ersten. Diese
schließt keine vollständige Runde ab, da das Fahrzeug vor der Linie startet;
der Bonus wird dennoch gewährt, weil das Erreichen der Linie auch aus einer
Teilrunde heraus das gewünschte Verhalten darstellt.

Die **Rundenzeitverbesserung** belohnt das Unterbieten der bisher besten Runde
derselben Episode, proportional zur Zahl der eingesparten Frames. Die erste
vollständige Runde setzt lediglich die Referenz und erzeugt keinen Bonus, da
eine Verbesserung gegenüber einem unendlichen Ausgangswert nicht definiert ist.
Die Boni teleskopieren: ihre Summe über eine Episode beträgt stets
`(erste Runde − beste Runde) · 5.0`, unabhängig davon, wie sich die Verbesserung
über die Episode verteilt hat. Der Bezugswert wird zu Beginn jeder Episode
zurückgesetzt, die Bestzeit ist also episodenlokal.

Bemerkenswert ist die Größenordnung: eine Verbesserung um 20 Frames entspricht
+100 und damit dem doppelten Rundenbonus. Der Term ist trotz seines kleinen
Faktors nicht nachrangig.

### 4.2.3 Abbruchbedingungen

Eine Episode endet auf drei Wegen, die sich in ihrer Behandlung durch den
Lernalgorithmus unterscheiden:

| Bedingung | Schwelle | Zusatzstrafe | Signal |
|---|---|---|---|
| Crash | Kontakt mit der Begrenzung | −300 | `terminated` |
| Reward-Boden | kumulierter Episodenreward ≤ −3000 | −100 | `terminated` |
| Zeitlimit | 2000 Frames | keine | `truncated` |

Die Unterscheidung zwischen `terminated` und `truncated` ist für das Lernen
wesentlich. `terminated` bezeichnet einen echten Endzustand des
Entscheidungsprozesses: nach ihm existiert keine Zukunft, der Wert des Zustands
ist vollständig durch die erhaltene Belohnung bestimmt. `truncated` bezeichnet
dagegen einen künstlichen Schnitt durch den Trainingsaufbau — das Fahrzeug hätte
weiterfahren können. Der Algorithmus muss über diese Grenze hinweg
bootstrappen, andernfalls lernt der Kritiker, ein erfolgreich zu Ende gefahrener
Lauf sei am Ende wertlos.

Der **Reward-Boden** ist eine Notbremse gegen aussichtslose Episoden. Ohne ihn
würde ein Agent, der weder vorankommt noch verunfallt, die vollen 2000 Frames
an Rechenzeit verbrauchen. Ein offener Befund hierzu findet sich in
Abschnitt 4.4.3.

### 4.2.4 Im Code vorhandene, aber nicht erreichbare Bestandteile

Vier Bestandteile sind implementiert, können unter der aktuellen Fahrzeugphysik
und Streckengeometrie jedoch nicht auslösen. Sie werden hier aufgeführt, weil
sie in der Konfiguration erscheinen und andernfalls fälschlich als wirksam
gelesen würden. Der Nachweis findet sich in Abschnitt 4.4.1.

| Bestandteil | Wert | Grund der Unerreichbarkeit |
|---|---|---|
| Falschrichtung | −30 pro Frame, zusätzlich `−v · 5.0` | Streckengeometrie |
| Rückwärtsfahren | `+v · w_reverse`, `w_reverse = 10.0` | Fahrzeugmodell erzwingt `v ≥ 0` |
| Rückwärtsüberquerung der Ziellinie | −50 | setzt `v < 0` voraus |
| Oszillationsschutz | `w_integral = 0.0` | bewusst deaktiviert, siehe 4.3.3 |

Die Falschrichtungsstrafe leitet sich nicht aus dem Vorzeichen der
Geschwindigkeit ab, sondern aus dem Vergleich der Fahrzeugausrichtung mit der
lokalen Streckenrichtung. Sie greift, wenn die Abweichung 90° überschreitet.
Bei aktiver Strafe entfällt zusätzlich der gesamte Geschwindigkeitsterm, da
beide Zweige einer Fallunterscheidung bilden.

### 4.2.5 Größenordnungen

Die relative Gewichtung der Bestandteile erschließt sich nicht aus den
Parameterwerten allein. Für einen erfolgreichen Lauf über die vollen 2000 Frames
bei durchschnittlich 1.2 m/s ergibt sich näherungsweise:

```
Geschwindigkeit    2000 Frames × 2.88     ≈  +5760
Rundenboni         9 Runden × 50          ≈   +450
Rundenzeitbonus    typisch                ≈   +100
Crash am Ende                                 −300
```

Der Geschwindigkeitsterm dominiert um eine Größenordnung und trägt rund 90 %
des Gesamtreturns. Die Crashstrafe entspricht etwa 100 Frames guter Fahrt, also
gut drei Sekunden; der Rundenbonus entspricht 17 Frames. Die Belohnungsfunktion
ist damit im Kern ein Geschwindigkeitssignal, das durch zielbezogene Ereignisse
ergänzt wird — eine Kalibrierung, deren Angemessenheit in Abschnitt 4.3.4
geprüft wird.

## 4.3 Entwicklung der Belohnungsfunktion

Die in Abschnitt 4.2 beschriebene Fassung ist das Ergebnis mehrerer Korrekturen.
Die folgenden Abschnitte stellen nicht deren zeitlichen Ablauf dar, sondern
ordnen sie nach dem jeweils zugrundeliegenden Fehlermuster, da diese Muster über
den Einzelfall hinaus übertragbar sind. Jeder Abschnitt folgt derselben
Gliederung: beobachtetes Symptom, Diagnose, vorgenommene Änderung.

### 4.3.1 Strafen müssen teleskopieren wie Boni

Eine frühere Fassung verrechnete jede gefahrene Runde gegen die bisherige
Bestzeit der Episode und vergab dabei sowohl einen Bonus für Verbesserungen als
auch eine Strafe für Verschlechterungen, letztere mit dem Faktor 2.0 pro
verlorenem Frame.

Beide Terme verhalten sich jedoch strukturell verschieden. Die Boni
teleskopieren, ihre Summe ist stets die Differenz zwischen erster und bester
Runde. Die Strafen teleskopieren nicht: sie fallen bei jeder einzelnen Runde
erneut an, die langsamer ist als die bisherige Bestzeit. Eine einzelne schnelle
Runde hebt damit die Bezugsmarke für alle folgenden Runden an und wird über den
Rest der Episode wiederholt bestraft:

```
sieben gleichmäßige Runden à 300 Frames             →     0
dieselben sieben, davon eine mit 250 Frames         →  −250
stetige Steigerung von 300 auf 240 Frames           →  +300
```

Schnelles Fahren war unter dieser Regel netto bestraft, sofern es nicht
gleichmäßig gesteigert wurde. Die Strafe wurde ersatzlos entfernt. Dass
langsames Fahren unerwünscht ist, regelt bereits der Geschwindigkeitsterm in
jedem einzelnen Frame; eine zweite, zustandsabhängige Kodierung derselben
Anforderung war nicht nur redundant, sondern gegenläufig.

**Übertragbare Regel:** Werden Bonus und Strafe gegen denselben veränderlichen
Bezugswert gerechnet, müssen beide dasselbe Teleskopierverhalten aufweisen.
Andernfalls entsteht eine Sperrklinke, die genau das Verhalten bestraft, das
belohnt werden soll.

### 4.3.2 Maßgeblich ist die Ableitung, nicht der Betrag

Mit der Umstellung auf das gemessene Fahrzeugmodell sank die
Anfahrbeschleunigung auf rund 0.25 m/s². Agenten verharrten daraufhin über
längere Phasen im Bereich zwischen 0.1 und 0.5 m/s, in den das Fahrzeug nach
jedem Bremsvorgang in einer Kurve gerät.

Die damals verwendete Stillstandsstrafe war als Klippe ausgeführt: sie betrug
−15 pro Frame, galt aber ausschließlich im Band `−0.1 ≤ v ≤ 0.1` und oberhalb
davon nicht mehr. Der Geschwindigkeitsterm existierte in diesem Bereich zwar,
war aufgrund seiner quadratischen Form jedoch numerisch bedeutungslos:

```
v = 0.1  →  +0.02        v = 0.5  →  +0.50
v = 0.2  →  +0.08        v = 1.2  →  +2.88
v = 0.3  →  +0.18        v = 1.5  →  +4.50
```

Bei 0.3 m/s beträgt die Belohnung für Fahren +0.18 pro Frame, während eine
einzelne Lenkbewegung bis zu −0.25 kostet. Das Nutzsignal lag damit unterhalb
des Rauschens der übrigen Terme.

Entscheidend ist dabei nicht der Betrag, sondern die Ableitung, denn sie
bestimmt, in welche Richtung sich die Strategie verschiebt:

```
d/dv (2v²) = 4v          bei v = 0.3  →  1.2 pro m/s
                         bei v = 1.2  →  4.8 pro m/s
```

Der Anreiz, schneller zu werden, ist genau dort am schwächsten, wo er am
dringendsten benötigt wird. Die Klippe verschärfte dies, da sie oberhalb von
0.1 m/s keinerlei Gradienten beitrug.

Sie wurde durch das in Abschnitt 4.2.1 beschriebene lineare Gefälle ersetzt.
Dessen Ableitung ist über den gesamten langsamen Bereich konstant:

```
d/dv (Gefälle) = w_slow / v_slow = 4.0 / 0.6 = 6.67 pro m/s
```

Das ist bei 0.3 m/s ein rund fünfeinhalbfach stärkeres Signal als der
quadratische Term. Der Bereich, in dem das Fahrzeug tatsächlich verharrte,
erhielt damit erstmals einen verwertbaren Gradienten.

**Übertragbare Regel:** Bei der Gestaltung einer Belohnungsfunktion ist nicht
maßgeblich, ob ein Term in einem Zustandsbereich vorhanden ist, sondern ob seine
Ableitung dort groß genug ist, um das Verhalten zu verschieben.

### 4.3.3 Die Metrik muss die beabsichtigte Größe messen

Neben der Lenkglättung war ein Oszillationsschutz vorgesehen, der schnelles Hin-
und Herlenken erfassen sollte — ein Verhalten, bei dem jede einzelne Änderung
klein bleibt und von der Lenkglättung daher unterbewertet wird. Er bestrafte die
Summe der Lenkänderungsbeträge über die letzten 30 Frames, soweit sie eine
Schwelle von 2.0 überstieg.

Die Zerlegung eines Laufs zeigte, dass er nicht das beabsichtigte Verhalten
erfasste, sondern jede Lenktätigkeit besteuerte. Ein gut fahrender
Referenzagent zahlte über 2000 Frames −625 und damit mehr, als seine gesamten
Rundenboni einbrachten (+600); der Mittelwert seiner Lenkänderungssumme lag bei
etwa 5.9 und damit weit über der Schwelle.

Die Ursache ist konstruktiv, nicht parametrisch. Beide Terme summieren Beträge
und verwerfen damit das Vorzeichen — genau jene Information, die Oszillation von
gerichtetem Lenken unterscheidet:

```
Oszillation      +0.2, −0.2, +0.2, −0.2, …    Σ|Δ| = 6.0    netto  0.0
gerichtete Kurve +0.2, +0.2, +0.2, +0.2, …    Σ|Δ| = 6.0    netto  6.0
```

Beide Fahrweisen erzeugen dieselbe Strafe. Der Oszillationsschutz ist damit
lediglich eine gefensterte und mit einem Freibetrag versehene Kopie der
Lenkglättung; keine Wahl der Schwelle hätte ihn funktionsfähig gemacht.

Der Term wurde durch Setzen von `w_integral = 0.0` deaktiviert statt entfernt,
um die Konstruktion für spätere Arbeiten nachvollziehbar zu halten. Ein
funktionsfähiger Detektor müsste stattdessen das Verhältnis `Σ|Δ| / |ΣΔ|`
auswerten, das für gerichtetes Lenken bei eins liegt und für reine Oszillation
divergiert, oder die Zahl der Vorzeichenwechsel im Fenster zählen.

**Übertragbare Regel:** Vor der Kalibrierung eines Terms ist zu prüfen, ob die
gewählte Kenngröße das beabsichtigte Verhalten überhaupt von zulässigem
Verhalten unterscheiden kann.

### 4.3.4 Kalibrierung der Crashstrafe

Wie in Abschnitt 4.2.5 dargestellt, entspricht die Crashstrafe von −300 nur etwa
100 Frames guter Fahrt. Hiergegen lässt sich einwenden, eine derart milde Strafe
könne absichtliches Verunfallen als Ausweg aus ungünstigen Zuständen attraktiv
machen: befindet sich der Agent bei niedriger Geschwindigkeit und sammelt
fortlaufend negative Rewards, erschiene der sofortige Abbruch günstiger als das
Weiterfahren.

Diese Überlegung setzt jedoch voraus, dass eine Erholung nicht möglich ist. Bei
0.25 m/s² dauert die Beschleunigung von 0.2 auf 1.2 m/s rund vier Sekunden
beziehungsweise 120 Frames:

```
Kosten während der Erholung                    ≈  −120
danach 1380 Frames à +2.88                     ≈ +3974
                                                  ──────
Erholung                                          +3850
Crash                                              −300
```

Die Erholung ist damit um etwa 4000 Rewardeinheiten günstiger als der Abbruch.
Der behauptete Fehlanreiz besteht nicht.

Empirisch bestätigt dies das gemessene Verhalten im langsamen Zustand. Erfasst
wurden die mittleren Stellwerte für Gas und Bremse, getrennt nach
Geschwindigkeit, ab Frame 30 zur Ausklammerung des Anfahrvorgangs, aus zufällig
gestreuten Startpositionen über je 20 Episoden:

| Agent | langsam (v < 0.5) Gas / Bremse | schnell Gas / Bremse | Crashquote |
|---|---|---|---|
| Lidar-Referenzagent | +1.000 / +0.000 | +0.570 / +0.281 | 0 / 20 |
| mit gestreuten Starts trainiert | +0.903 / +0.217 | +0.626 / +0.427 | 19 / 20 |
| mit festem Start trainiert | +0.178 / +0.518 | +0.603 / +0.417 | 10 / 20 |

Beide Agenten, die diesen Zustandsbereich im Training gesehen haben, geben bei
niedriger Geschwindigkeit mehr Gas als bei hoher. Nur der von einem festen
Startpunkt trainierte Agent bremst dort.

Der Vergleich der letzten beiden Zeilen ist entscheidend, da beide Agenten
derselben Belohnungsfunktion unterliegen. Prämierte diese das Aufgeben, hätte
auch der mit gestreuten Starts trainierte Agent dieses Verhalten erlernen
müssen; er hat das Gegenteil erlernt. Das Bremsen des dritten Agenten ist damit
auf *covariate shift* zurückzuführen — er trifft auf Zustände außerhalb seiner
Trainingsverteilung — und nicht auf eine Fehlkalibrierung der Belohnung.

Die milde Crashstrafe ist damit als bewusste Entscheidung zu werten. Für eine
Rennaufgabe ist sie die angemessenere Wahl: eine hohe Crashstrafe erzeugt einen
Agenten, der mit geringer Geschwindigkeit fährt und nie verunfallt, was im
Reward gut abschneidet, die Aufgabenstellung aber verfehlt. Übervorsichtigkeit
ist bei Aufgaben dieses Typs der häufigere Entwurfsfehler als das Gegenteil.

Der Fehlanreiz bliebe allerdings dann bestehen, wenn eine Erholung teuer würde —
bei trägerer Beschleunigung, längeren Episoden oder einer Streckengeometrie, in
der sich das Fahrzeug festfahren kann. Für die Übertragung auf die
Simulationsumgebung ist dies zu prüfen und nicht ungeprüft zu übernehmen.

### 4.3.5 Konfiguration und Wirkung müssen übereinstimmen

Zwei Parameter, `best_lap_bonus` mit dem Wert 200.0 und `slow_lap_penalty` mit
dem Wert −30.0, waren in jeder gespeicherten Trainingskonfiguration enthalten,
wurden vom Code jedoch nie ausgelesen. Die tatsächlich wirksamen Faktoren waren
hartkodierte Werte von 5.0 beziehungsweise 2.0.

Der Unterschied betrifft nicht nur den Betrag, sondern die Form der Belohnung.
Die dokumentierten Parameter hätten eine pauschale Vergabe bewirkt, unabhängig
vom Ausmaß der Verbesserung; die tatsächlich implementierte Rechnung war
proportional zur Zahl der eingesparten Frames. Eine Pauschale von 200 hätte eine
Verbesserung um einen einzelnen Frame ebenso hoch vergütet wie eine um zwanzig
und damit eine grundlegend andere Anreizstruktur erzeugt.

Für die Auswertung folgt daraus, dass sämtliche Konfigurationen vor der Version
`v3_rundenzeit` zwei wirkungslose Zahlen enthalten. Ein Vergleich von
Trainingsläufen auf Grundlage dieser Dateien beschreibt nicht die tatsächlich
verwendete Belohnungsfunktion.

Der umgekehrte Fall besteht ebenfalls. Drei Zahlen wirken auf die Belohnung,
erscheinen aber in keiner Konfigurationsdatei, da sie außerhalb der
serialisierten Klasse liegen: der Faktor 5.0 der Zusatzstrafe bei
Falschrichtung sowie Schwelle (−3000) und Zusatzstrafe (−100) des Reward-Bodens.

**Übertragbare Regel:** Jeder wirksame Faktor gehört als benannter Parameter in
die serialisierte Konfiguration, und jeder dort enthaltene Parameter muss
tatsächlich ausgelesen werden. Andernfalls dokumentiert die Konfiguration einen
anderen Versuchsaufbau als den durchgeführten.

### 4.3.6 Fehlerhafte Rundenzeitmessung

Die Rundenzeitverbesserung nach Abschnitt 4.2.2 blieb über einen Zeitraum von
rund drei Wochen wirkungslos. Ursache war die Behandlung der ersten
Ziellinienüberquerung nach einem Reset: Das Fahrzeug startet vor der Linie, die
erste Überquerung schließt daher keine vollständige Runde ab. Sie wurde dennoch
als Rundenzeit gewertet, wodurch die Referenz auf etwa zehn Frames gesetzt
wurde. Eine reguläre Runde benötigt rund 180 Frames und konnte diesen Wert
folglich nie unterbieten.

Die Korrektur führt eine gesonderte Behandlung der ersten Überquerung ein: Sie
startet die Zeitmessung, wird selbst aber nicht als Rundenzeit gewertet. Der
Rundenbonus nach Abschnitt 4.2.2 wird weiterhin vergeben.

Der Fehler ist insofern instruktiv, als er über den gesamten Zeitraum ohne
Fehlermeldung bestand und sich in keiner Kennzahl niederschlug — die
Belohnungsfunktion arbeitete fehlerfrei, lediglich ein Term war faktisch
abgeschaltet. Entdeckt wurde er erst durch die systematische Zerlegung eines
Laufs in seine Rewardbestandteile.

## 4.4 Validierung der Belohnungsfunktion

Die vorstehenden Abschnitte beschreiben, welche Bestandteile die
Belohnungsfunktion enthält. Offen bleibt, welche davon im laufenden Betrieb
tatsächlich auslösen. Diese Frage wird üblicherweise nicht gestellt; sie ist
jedoch notwendig, da ein nie ausgelöster Term zwar dokumentiert wird, aber
keinerlei Wirkung auf das gelernte Verhalten entfaltet.

### 4.4.1 Nicht auslösende Bestandteile

Gezählt wurde über vollständige Episoden aus zufällig gestreuten
Startpositionen, wie viele Frames die Falschrichtungsbedingung erfüllen:

```
trainierter Agent      0 von 20000 Frames    =  0.00 %
zufällige Steuerung    0 von  3471 Frames    =  0.00 %
```

Bemerkenswert ist der zweite Wert: auch eine rein zufällige Steuerung dreht das
Fahrzeug nie um mehr als 90° aus der Streckenrichtung. Die Ursache ist
geometrisch. Bei einer verbleibenden Fahrbahnbreite von ±0.127 m und einem
maximalen Lenkwinkel von 30° erreicht das Fahrzeug die Begrenzung, bevor die
erforderliche Drehung zustande kommt.

Die Falschrichtungsstrafe ist damit unter der gegebenen Streckengeometrie
unerreichbar, ebenso die von ihr abgeleitete Zusatzstrafe. Gleiches gilt aus
physikalischen Gründen für die Rückwärtsfahrstrafe und die Strafe für
rückwärtiges Überqueren der Ziellinie, da das Fahrzeugmodell negative
Geschwindigkeiten ausschließt.

Für eine Übertragung auf eine andere Umgebung ist dies erneut zu prüfen: sobald
die Fahrbahn breiter ist oder Rückwärtsfahren ermöglicht wird, werden diese
Fälle erreichbar, und die zugehörigen Gewichte wären erstmals wirksam, ohne je
erprobt worden zu sein.

### 4.4.2 Zur Methodik der Prüfung

Eine erste Erhebung der Abbruchgründe ergab, dass der Reward-Boden nie auslöst:

```
                Crash   Reward-Boden   Zeitlimit
trainiert           0              0          30
zufällig           30              0           0
```

Dieser Schluss war falsch, und zwar aufgrund einer ungeeigneten Stichprobe.
Beide geprüften Fälle können den Boden konstruktionsbedingt nicht erreichen: der
trainierte Agent sammelt keine negativen Rewards, die zufällige Steuerung
verunfallt nach durchschnittlich 350 Frames und damit lange vor Erreichen der
Schwelle.

Die Wiederholung mit einem *schwachen* Agenten — trainiert von einem festen
Startpunkt, bewertet aus gestreuten Startpositionen und damit außerhalb seiner
Trainingsverteilung — ergibt ein anderes Bild:

```
Crash 11    Reward-Boden 9    Zeitlimit 0
mittlere Episodenlänge 571, kürzeste 47, längste 1516
```

Der Boden löst in neun von zwanzig Episoden aus. Er greift im Zwischenfall:
schlecht genug für dauerhaft negative Rewards, aber nicht schlecht genug für
einen sofortigen Unfall. In der frühen Trainingsphase ist dies der Regelfall.

**Methodische Regel:** Der Nachweis, dass ein Bestandteil nie auslöst, verlangt
eine Stichprobe, die den Zustandsbereich abdeckt, in dem er auslösen *kann*. Ein
gut trainierter Agent und eine zufällige Steuerung sind hierfür beide
ungeeignet — der eine erreicht den kritischen Bereich nicht, der andere
durchläuft ihn zu schnell.

### 4.4.3 Offener Befund: Behandlung des Reward-Bodens

Der Reward-Boden setzt das Signal `terminated`. Nach der in Abschnitt 4.2.3
dargestellten Unterscheidung ist dies unzutreffend: Der Abbruch erfolgt durch
den Trainingsaufbau, nicht durch einen Endzustand des Entscheidungsprozesses.
Das Fahrzeug hätte weiterfahren können. Der Kritiker lernt somit, der Wert des
betroffenen Zustands sei vollständig durch die Zusatzstrafe von −100 bestimmt,
was die Wertschätzung in diesem Zustandsbereich systematisch verzerrt.

Es handelt sich um dieselbe Fehlerklasse, die für das Zeitlimit bereits behoben
wurde. Betroffen sind ausschließlich schwache Agenten und damit die frühe
Trainingsphase sowie erschwerte Startbedingungen. Ein Zusammenhang mit dem
beobachteten Leistungsabfall fortgeschrittener Trainingsläufe ist naheliegend,
aber nicht nachgewiesen.

## 4.5 Zusammenfassung

Die Belohnungsfunktion ist im Kern ein quadratisches Geschwindigkeitssignal, das
durch einen Rundenbonus, eine Rundenzeitkomponente und zwei regularisierende
Strafterme ergänzt wird. Der Geschwindigkeitsterm trägt rund 90 % des
Gesamtreturns.

Aus ihrer Entwicklung ergeben sich vier übertragbare Regeln:

1. Bonus und Strafe gegen einen veränderlichen Bezugswert müssen dasselbe
   Teleskopierverhalten aufweisen (4.3.1).
2. Maßgeblich für die Wirksamkeit eines Terms ist seine Ableitung im relevanten
   Zustandsbereich, nicht sein Betrag (4.3.2).
3. Eine Kenngröße ist vor ihrer Kalibrierung darauf zu prüfen, ob sie das
   beabsichtigte Verhalten überhaupt unterscheiden kann (4.3.3).
4. Konfiguration und wirksame Faktoren müssen deckungsgleich sein, in beide
   Richtungen (4.3.5).

Die Validierung nach Abschnitt 4.4 zeigt darüber hinaus, dass vier der
implementierten Bestandteile unter den gegebenen Bedingungen nie auslösen. Die
Belohnungsfunktion ist damit einfacher, als ihre Implementierung nahelegt — ein
Umstand, der ohne gezielte Prüfung unbemerkt geblieben wäre.
