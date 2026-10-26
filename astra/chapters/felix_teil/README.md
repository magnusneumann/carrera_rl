# Felix' Teil (Phase 2: 3D-Umsetzung in Isaac)

LaTeX-Fassung von `Doku/Arbeit V14 - Claude.docx` (Stand 26.10.2026), im Stil von Magnus' Kapiteln.
Magnus' Dateien (`main.tex`, `bibliography.bib`, `chapters/0*.tex`) sind **nicht** verändert.

## Einbinden

Ist in `main.tex` bereits eingetragen (26.10.2026), hinter `\include{chapters/03-hauptteil}`:

```latex
\include{chapters/felix_teil/05-umsetzung-3d}
\include{chapters/felix_teil/06-trainingsablauf}
```

und bei den Literaturquellen:

```latex
\addbibresource{chapters/felix_teil/felix_literatur.bib}
```

Kapitel-, Abbildungs- und Tabellennummern vergibt LaTeX selbst. Im Text stehen nur `\ref{...}`.

## Inhalt

| Datei | Inhalt |
|---|---|
| `05-umsetzung-3d.tex` | Kapitel „Umsetzung in 3D“ (Word: Kapitel 5) |
| `06-trainingsablauf.tex` | Kapitel „Trainingsablauf in Isaac“ (Word: Kapitel 6) |
| `felix_literatur.bib` | Tobin 2017, Yarats 2021 (SAC-AE), Sketchfab-Modell, skrl, Isaac Lab |
| `bilder/` | alle Abbildungen in Originalauflösung |

## Abbildungen

| Word | Label | Datei |
|---|---|---|
| 5.1 | `fig:3d_fahrzeugmodell` | `fahrzeugmodell_isaac.png` |
| 5.2 | `fig:3d_fahrzeug_real` | `fahrzeug_real.jpg` |
| 5.3 | `fig:3d_lackierung` | `fahrzeug_ohne_mit_lackierung.png` |
| 5.4 | `fig:3d_grenzen_isaac` | `grenzen_kontrollanzeige.jpg` |
| 5.5 | `fig:3d_strecke_grenzen` | `strecke_grenzen.png` |
| 6.1 | `fig:pipeline_stufen` | `pipeline_stufen.png` |
| 6.2 | `fig:gegenversuch_einfrieren` | `gegenversuch_einfrieren.png` |
| 6.3 | `fig:agent1_verlauf` | `agent1_verlauf.png` |
| 6.4 | `fig:darstellungen` | `darstellungen.png` |
| 6.5 | `fig:agent2_verlauf` | `agent2_verlauf.png` |
| 6.6 | `fig:agent2_darstellungen` | `agent2_darstellungen.png` |
| 6.7 | `fig:transfer` | `transfer.png` (Legende jetzt „Weitertraining/Neustart“ statt „warm/kalt“) |

Tabellen: `tab:3d_hardware`, `tab:3d_fahrzeugdaten`, `tab:3d_belohnung`, `tab:gegenversuch_einfrieren`,
`tab:agent1_einstellungen`, `tab:agentenvergleich`, `tab:ohne_nachtraining`, `tab:transfer`.

## Offen (im Quelltext als `% TODO` markiert)

- Abschnitte „Was übernommen wird“ und „Was geändert werden musste“ sind leer.
- Stufe 2: Datenquelle von Agent 2 fehlt.
- Verweis auf den Abschnitt „Einschränkungen“ (Kapitel existiert noch nicht).
- Quelle des Produktbilds (`fahrzeug_real.jpg`) und Autor/Lizenz des Sketchfab-Modells.
- Ungetestet kompiliert: Auf Felix' Rechner ist kein LaTeX installiert. Syntax, Labels, Zitate,
  Abkürzungen und Bildpfade wurden per Skript geprüft.
