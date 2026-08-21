schlecht trainierter agent für demo
demo muss mehr varianz haben - mensch inkput
Head freezen nach vortraining? Vortraining daten - wie viel gut?



# Struktur des Projektberichts

Doku soll auf digitalen gebrauch optimiert sein - HKA_logo.png obenrechts - seitanzahl unten echts - linie die Kopf und Fußzeile abtrennen.
Besonders wichtig:
Die wichitgen schlüsse - vortraining wirkt - nur der stack funktioniert gut genug 
Konzept entwicklung - dann notebook vorgehen um zu validieren und zu verbessern. das gesamtvorgehen muss genau in der dokumetation vorliegen - und beweise die aktuell fehlen müssen alle nachträglich erstellt werden (training nochmal - eval nochmal usw. für Belege) Fehlstellen festhalten!! Dazu bitte eine doku_tasks.md anlegen - welche offene Kapitel und offene Beweise trackt.


Deckblatt
    Inhalt -Fakultät für Maschinenbau und Mechatronik
            Forschungs- und Entwicklungsprojekt
            Studiengang Robotik und KI in der Produktion
            Titelplatzhalter
            Vorgelegt von: Felix Faaß, Magnus Neumann
            Matrikelnummer: [Platzhalter]
            Hochschule Kalrsruhe
            Projektbetreuer Prof. Dr. Björn Hein

            Datum der Abgabe Platzhalter
Eigenständigkeitserklärung
Kurzfassung
Abstract auf Englisch
Inhaltsverzeichnis
    Einleitung
        Motivation - CarreraHybrid (RC Autos), RL anwenden - Vision Agent kann mit Kamera an der Decke Autos auf Strecken präziese steuern.
        
        Projekt Struktur - Diese Projektarbeit behandelt die erste Hälfte der Vision. Dafür wurden RL Grundlagen und ähnliche Experimente aus Papern analysiert um vielversprechende Architekturen und Konzepte in einer 2D Umgebung zeiteffizient zu testen. Im Anschluss wird der beste Ansatz in 2D Validiert um das Verfahren in 3D SIM (Isaac Sim) zu anzuwenden. Ein Folgeprojekt könnte die Erforschung der Sim to Real Gap sein mit dem Ziel das vorhandene CarerraHybrid Auto von einem Agenten auf einer realen Strecke fahren zulassen.

        Ziele - Agent der auf der Strecke bestzeiten fährt - die Strecken markierungen achtet - den menschen und andere einfachere Agenten schlägt - das gesamt Bild der Strecke mit auto verarbeitet (Vision Action/physical AI artig) das was auch ein Mensch als input für entscheidungen nutzt.

    Theoretische Grundlagen
        Hier alle Grundlagen mit Quellen erklären - Quellen in Quellenverzeichnis am Ende der md auflisten. Kann man quellen oder Kapitel oder Abschnitte in .md verlinken? Bleibt der Link wenn in eine pdf umgewandelt wird?
        In documentation.md sind wichtige Ansätze und Paper bereits genannt.
    Hauptteil
        Projekt Verlauf erklären. Welche Ansätze schienen in den Papern interessant -welche Inspiration wir auf zB Youtube entdeckt (trackmania weltrekorde, und yt Kanal Gonkee)
        Vorstellungen der 2D Konzepte
        
        Technische Umsetzung des Testsetups um Konzepte zu testen - Das was im Notebook_protoryping steht chronologisch erklären. und auf das notebook verweisen - spannendere Arbeitsdokumentation die interaktiv ist zum selber klicken.

        Erkenntnisse aus den Tests - Ergebnisse in Chronologischer reihenfolge die dazu zum besten Ansatz führen
        Multi Stacked vision training mit pretraining from demostration - Alle erkenntisse aus erkenntnisse.md aufzeigen - das sind die feinheiten die wichtig sind - was wurde probiert und was hat nicht funktioniert? zb stacked vision mit dilatation aktuelles bild 3. letztes 7. letztes (oder so ähnlich)

        Übertragung in die 3D Simulation
        Wie ist das aufgebaut Platzhalter

        Funktioniert das in 3D?

        Ergebnis - finale bewertung des 3d RL agenten - werden die erwartungen erfüllt?

    Schluss
        Fazit
        learnings
        Ausblick für Folgenprojekt
    
    Quellenverzeichnis
    ggf ABbildungs/tabellenverzeichnis
    Anhang
        
# Struktr des Projektberichts Ende

Ein Notebook dokumentiert den Arbeits/Denk/Forschungsprozess und dient als Interaktive Dokumentation mit welcher der Entscheidungsprozess und die Simulation zum mit machen vorgeführt wird.
Das Auto brauch eine Trägheit und auch eine Art Grip/Höchstgeschwindigkeit, ggf hardgecoded einen Schleudervorgang, bei dem das Fahrzeug ausbricht in die aktuelle Lenkrichtung und die letzte Trajektorie beibehält.
Das Auto sollte etwas Trägheit haben und weiter rollen auch wenn der Gasbefehl grade nicht aktiv ist.

Notebook ablauf:
Ziel erklären
Umgebung mit Auto
Physik Grip, Trägheit, Lenken mit Geschwindigkeit andere Wirkung?
Steuerungbauen, WASD
Pi, die Policy definieren nicht auf Rand/ weiß fahren, nicht rückwärts, vorwärts gut, bei besserer(höherer) oder gleicher rundenzeit reward. je höher v desto höher reward, wenige wechsel zwischen gas bremse und lenkung reward (smooth fahren)
Virtuelle Kamera/CV damit man checken kann wo und wohin das auto ist/will?
Modell konfigurieren
Training configurieren
training starten
weitere trainings/arichtekturen/modelle konfigurieren,
funktion/schliefe die alle trainings/architekturen lernen lässt und testet/vergleicht


Ich will Modell-frei PPO
mit parametern/raycasts - müsste in real dann aus opencv kommen oder ähnlichem
geschwindigkeit lenkwinkel aus ausgabe und bild
nur 1 bild pro step
2-3 bilder 

Ich will Modell-based probieren
Ich will Imitation/Demonstration Learning probieren

Virtuelle Kamera Bild - Reward funktion wie?
Was für ein Netz ist aktuell hinten dran. Muss das größer werden bei mehr Input?
Agenten weiter trainieren

Mit Bilddaten lernen
TrainingsFactory
