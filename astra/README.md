# Academic Scientific Thesis & Report Architecture

**ASTRA** (**A**cademic **S**cientific **T**hesis & **R**eport **A**rchitecture)  ist ein modernes, modular aufgebautes LaTeX-Template für wissenschaftliche Arbeiten wie Bachelor- und Masterarbeiten. Es ist speziell auf die Anforderungen der Hochschule Karlsruhe (HKA) zugeschnitten (z.B. Logos, Struktur, Seitengestaltung), lässt sich aber mit wenigen Anpassungen auf andere Hochschulen, Fakultäten und Studiengänge übertragen. Das Template bietet eine saubere Struktur, vordefinierte Formatierungen, Beispiele für Abbildungen, Tabellen, Formeln und Code-Listings sowie integrierte Unterstützung für Biber/BibLaTeX, Abkürzungsverzeichnis und Nomenklatur.

Diese LaTeX-Vorlage ist speziell für die Erstellung wissenschaftlicher Arbeiten wie Master- oder Bachelorarbeiten konzipiert. Sie bietet eine klare Struktur, vordefinierte Formatierungen und zahlreiche Beispiele für gängige Elemente wie Abbildungen, Tabellen, Formeln, Code-Listings und Literaturnachweise, um einen schnellen Start zu ermöglichen.

## 1. Voraussetzungen

Um diese Vorlage nutzen und kompilieren zu können, benötigst du eine vollständige LaTeX-Distribution und einige Hilfsprogramme:

-   **LaTeX-Distribution:**
    *   **TeX Live** (empfohlen für Linux/macOS)
    *   **MiKTeX** (empfohlen für Windows)
    Diese Distributionen enthalten den `pdflatex`-Compiler und die meisten benötigten Pakete.
-   **Biber:** Für die Literaturverwaltung ist Biber notwendig. Es ist in der Regel Bestandteil der oben genannten LaTeX-Distributionen.
-   **`make` (empfohlen):** Zur Ausführung des beiliegenden Makefiles. Unter Linux/macOS standardmäßig vorhanden, unter Windows z.B. über WSL, MSYS2 oder Git Bash verfügbar.

### Benötigte LaTeX-Pakete
Die Vorlage lädt die wichtigsten Pakete über die Datei `settings.sty`. Stelle sicher, dass diese Pakete in deiner LaTeX-Installation vorhanden sind (was bei einer Vollinstallation der Distribution der Fall sein sollte):
*   `acronym`: Für Abkürzungsverzeichnisse.
*   `pgfplots`: Für hochwertige Diagramme direkt in LaTeX (siehe Beispiel in `chapters/05-implementation.tex`).
*   `listings`: Zum Einbinden und Formatieren von Quellcode (z.B. Python-Code in `chapters/04-methodology.tex`).
*   `booktabs`: Für professionell aussehende Tabellen.
*   `import`: Zum modularen Einbinden von `.tex`-Dateien aus Unterverzeichnissen (z.B. PGFPlots-Code oder Kapitel).
*   `subcaption`: Für das Erstellen von Unterabbildungen (Subfigures) mit eigenen Beschriftungen (siehe `chapters/06-evaluation.tex`).
*   `nomencl`: Für das Erstellen eines Nomenklatur- oder Symbolverzeichnisses.

## 2. Vorbereitung vor dem ersten Kompilieren

Bevor du die Arbeit das erste Mal kompilierst, solltest du folgende Punkte anpassen:

1.  **Metadaten in `settings.sty` anpassen:**
    Öffne `settings.sty` und passe die `\newcommand`s an deine Daten an. Dies beinhaltet:
    *   `\Title`, `\Subtitle`
    *   `\Name`, `\Id`, `\DateOfBirth`, `\PlaceOfBirth`
    *   `\Location` (Ort für die Eigenständigkeitserklärung)
    *   `\DeliveryDate` (Abgabedatum für die Eigenständigkeitserklärung)
    *   `\University`, `\Faculty`, `\Degree`, `\DegreeShort`
    *   `\DocName` (Kann z.B. auf `Bachelorarbeit` geändert werden)
    *   `\Supervisor`, `\CoSupervisor`
    *   `\CompanyName`, `\CompSupervisor` (Kommentiere `\CompanyName` aus oder setze es auf `NoCompanyInvolved`, wenn keine Firma involviert ist, um das Firmenlogo in der Kopfzeile auszublenden.)

2.  **Logos und Bilder:**
    Lege deine Universitäts- und ggf. Firmenlogos (z.B. `hka_logo.png`) im `figures/png/`-Ordner ab und passe die Dateinamen in `settings.sty` an. Die Beispielbilder `example-image-a` und `example-image-b` sollten durch deine eigenen Grafiken ersetzt werden.

3.  **Literatur (`bibliography.bib`):**
    Füge deine Literaturzitate im BibLaTeX-Format in die Datei `bibliography.bib` ein. Die Vorlage nutzt den IEEE-Zitierstil.

4.  **Abkürzungsverzeichnis (`acronyms.tex`):**
    Definiere deine Abkürzungen in `acronyms.tex` im Format `\acro{AI}{Künstliche Intelligenz}`. Sie werden dann automatisch im Text bei der ersten Verwendung ausgeschrieben und im Verzeichnis gesammelt.

5.  **Nomenklatur (`nomenclature.tex`):**
    Füge Symbole und Einheiten im `nomenclature.tex` ein.

## 3. Kompilieren der Arbeit

### 3.1 Empfohlener Weg: Mit dem beiliegenden Makefile
Die Vorlage bringt ein fertiges `Makefile` mit, das den gesamten Kompilierungsprozess automatisiert. Öffne ein Terminal im Hauptverzeichnis des Projekts und nutze einen der folgenden Befehle:

-   `make` (oder `make full`): Führt einen vollständigen Build durch (`pdflatex` + `biber` + Nomenklatur + 2x `pdflatex`). Dies garantiert, dass alle Referenzen, Verzeichnisse und die Literatur aktuell sind.
-   `make fast`: Führt nur einen schnellen `pdflatex`-Lauf durch. Ideal für reine Textänderungen, bei denen sich keine Referenzen geändert haben.
-   `make biber`: Aktualisiert gezielt die Literatur (inkl. Vor- und Nachlauf von `pdflatex`).
-   `make nomen`: Aktualisiert gezielt das Nomenklaturverzeichnis (inkl. Vor- und Nachlauf von `pdflatex`).
-   `make clean`: Löscht alle temporären Hilfsdateien (`.aux`, `.log`, `.toc`, etc.), behält aber das PDF.
-   `make fclean`: Löscht alle Hilfsdateien sowie das generierte PDF-Dokument (`main.pdf`).

### 3.2 Alternativer Weg: Manueller Workflow
Wenn du `make` nicht nutzen möchtest oder kannst, musst du die einzelnen Kompilierungsschritte manuell ausführen. Dies ist notwendig, da beispielsweise Inhaltsverzeichnis, Referenzen, Abkürzungsverzeichnis und Literaturverzeichnis mehrere Durchläufe benötigen, um korrekt aufgelöst zu werden.

1.  **Erster `pdflatex`-Lauf:** Erzeugt die `.aux`-Dateien, die für Biber und die Verzeichnisse benötigt werden.
    ```bash
    pdflatex main.tex
    ```

2.  **`biber`-Lauf:** Verarbeitet die Literatur (`bibliography.bib`) und erstellt die `.bbl`-Datei.
    ```bash
    biber main
    ```

3.  **Nomenklatur-Lauf:** Verarbeitet die Nomenklatureinträge.
    ```bash
    makeindex main.nlo -s nomencl.ist -o main.nls
    ```

4.  **Zweiter und Dritter `pdflatex`-Lauf:** Bindet die generierten Dateien ein und aktualisiert Referenzen sowie Verzeichnisse vollständig.
    ```bash
    pdflatex main.tex
    pdflatex main.tex
    ```

## 4. Projektstruktur

Das Projekt ist modular aufgebaut, um die Übersichtlichkeit zu gewährleisten:

```
.
├── main.tex                 # Hauptdatei des Dokuments
├── settings.sty             # Globale Einstellungen, Paketdefinitionen, Metadaten
├── acronyms.tex             # Definitionen für das Abkürzungsverzeichnis
├── bibliography.bib         # Literaturdatenbank im BibLaTeX-Format
├── nomenclature.tex         # Definitionen für die Nomenklatur
├── chapters/
│   ├── 00-titlepage.tex     # Titelseite
│   ├── 00-declaration.tex   # Eigenständigkeitserklärung
│   ├── 02-introduction.tex  # Einführung
│   ├── 03-related-work.tex  # Stand der Technik
│   ├── 04-methodology.tex   # Methodik
│   ├── 05-implementation.tex# Implementierung
│   ├── 06-evaluation.tex    # Evaluation der Ergebnisse
│   ├── 07-conclusion.tex    # Fazit und Ausblick
│   ├── 00-preface.tex       # (Auskommentiertes) Vorwort/Danksagung
│   ├── appendix-a.tex       # Beispiel Anhang A
│   └── appendix-b.tex       # Beispiel Anhang B
├── code/
│   └── csv_loader.py        # Beispiel Python-Code für Listings
├── figures/
│   ├── png/                 # Für PNG-Bilder (z.B. Logos)
│   │   └── hka_logo.png
│   ├── pdf/                 # Für PDF-Bilder
│   ├── svg/                 # Für SVG-Bilder (ggf. vor Kompilierung zu PDF konvertieren)
│   └── pgf/                 # Ausgelagerter PGFPlots-Code
│       └── training_plot.pgf
└── README.md                # Diese Anleitung
```

## 5. Anpassungen & Erweiterungen

*   **Neue Kapitel hinzufügen:** Lege neue `.tex`-Dateien im `chapters/`-Verzeichnis an und binde sie mit `\include{chapters/deine_datei}` in `main.tex` ein.
*   **Styling & Pakete:** Alle globalen Einstellungen und Paketimporte werden in `settings.sty` verwaltet. Hier kannst du das Design anpassen oder weitere Pakete hinzufügen.
*   **Universitäts-spezifische Klasse:** Falls deine Hochschule eine eigene `.cls`-Datei bereitstellt (z.B. `hskarlsruhe.cls`), kannst du diese im Projekt ablegen (z.B. im `classes/`-Ordner) und die erste Zeile in `main.tex` entsprechend anpassen: `\documentclass[...]{hskarlsruhe}`.
*   **Beispielinhalte:** Die Kapitel sind bewusst mit vielen Beispielkonstrukten (Tabellen, Formeln, Code, Abbildungen, Subfigures, Zitate, Fußnoten, Querverweise) versehen, um die Nutzung zu demonstrieren. Lösche oder ersetze diese nach Bedarf.

---
Viel Erfolg beim Schreiben deiner wissenschaftlichen Arbeit! 🎓
