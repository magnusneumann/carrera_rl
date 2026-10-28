# Felix' Teil (Phase 2: 3D-Umsetzung in Isaac)

LaTeX-Fassung von `Doku/Arbeit V18.docx` (Stand 28.10.2026), im Stil von Magnus' Kapiteln.
Text, Zahlen und Beschriftungen folgen der Word-Fassung; offensichtliche Tippfehler und
Widersprüche zu den Abbildungen sind hier bereits korrigiert.

## Einbinden

Ist in `main.tex` eingetragen, hinter `\include{chapters/03-hauptteil}`:

```latex
\include{chapters/felix_teil/05-umsetzung-3d}
\include{chapters/felix_teil/06-trainingsablauf}
```

und bei den Literaturquellen:

```latex
\addbibresource{chapters/felix_teil/felix_literatur.bib}
```

Im Gesamtdokument werden die beiden Kapitel zu Kapitel 4 und 5 (Word: 5 und 6); Magnus' Ergebnisse
und Fazit rücken auf 6 und 7. Alle Nummern vergibt LaTeX, im Text stehen nur `\ref{...}`.

## Inhalt

| Datei | Inhalt |
|---|---|
| `05-umsetzung-3d.tex` | Kapitel „Umsetzung in 3D“ (Word: Kapitel 5) |
| `06-trainingsablauf.tex` | Kapitel „Trainingsablauf in Isaac“ (Word: Kapitel 6) |
| `felix_literatur.bib` | Tobin 2017, Sketchfab-Modell (CC BY 4.0), Isaac Lab |
| `bilder/` | alle Abbildungen in Originalauflösung |

## Abbildungen und Tabellen

| Word | LaTeX-Label | Datei |
|---|---|---|
| Abb. 5.1 | `fig:3d_fahrzeug` | `fahrzeug_modell_und_real.jpg` |
| Abb. 5.2 | `fig:3d_lackierung` | `fahrzeug_ohne_mit_lackierung.png` |
| Abb. 5.3 | `fig:3d_grenzen_isaac` | `grenzen_kontrollanzeige.jpg` |
| Abb. 5.4 | `fig:3d_strecke_grenzen` | `strecke_grenzen.png` |
| Abb. 6.1 | `fig:pipeline_stufen` | `pipeline_stufen.png` |
| Abb. 6.2 | `fig:gegenversuch_einfrieren` | `gegenversuch_einfrieren.png` |
| Abb. 6.3 | `fig:agent1_verlauf` | `agent1_verlauf.png` |
| Abb. 6.4 | `fig:darstellungen` | `darstellungen.png` |
| Abb. 6.5 | `fig:agent2_verlauf` | `agent2_verlauf.png` |
| Abb. 6.6 | `fig:agent2_darstellungen` | `agent2_darstellungen.png` |
| Abb. 6.7 | `fig:transfer` | `transfer.png` |
| Tab. 5.1–5.4 | `tab:3d_hardware`, `tab:3d_fahrzeugdaten`, `tab:3d_belohnung`, `tab:3d_vergleich2d` | |
| Tab. 6.1–6.5 | `tab:gegenversuch_einfrieren`, `tab:agent1_einstellungen`, `tab:agentenvergleich`, `tab:ohne_nachtraining`, `tab:transfer` | |

Die Diagramme entstehen mit `carrera_rl/abbildungen_arbeit.py` im 3D-Repository.

## Format (wie Magnus' Kapitel)

- `\section`/`\subsection` jeweils mit `\label` direkt darunter und einer Leerzeile davor und danach.
- Kleine fette Überschriften (`\paragraph{...}`) nur dort, wo keine echte Überschrift steht
  (Stufen der Pipeline, „Bewertung“, „Ursache des Durchbruchs“, Leitfragen in 6.6.3).
- Beschriftung der Tabellen oben, der Abbildungen unten, alle mit Punkt am Ende.
- Zahlen und Einheiten mit `siunitx` (`\num`, `\SI`), Abkürzungen mit `\ac{...}`.

## Offen (im Quelltext als `% TODO` markiert)

- Verweis auf den Abschnitt „Einschränkungen“ (Abschnitt existiert noch nicht).
- Abstimmung mit Magnus: sein Abschnitt 3.2 (Gerüst für Phase 2), Einleitung, Tabelle
  „Abnahmekriterien“ und „Fazit zu Phase 2“ kennen diesen Teil noch nicht (Einzelheiten im
  Kommentar zu Kapitel 6 in der Word-Datei).
- Mit Tectonic (XeTeX) testweise kompiliert, ohne Fehler und ohne offene Verweise; mit pdfLaTeX
  und biber noch nicht (auf Felix' Rechner nicht installiert).
