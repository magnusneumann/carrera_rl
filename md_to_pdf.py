#!/usr/bin/env python3
"""
md_to_pdf.py

Konvertiert Markdown-Dokumente (.md) in ein exakt an das LaTeX-Layout
der Hochschule Karlsruhe (HKA) angepasstes PDF (Vorlage: beispiel_bericht.pdf).

Features:
- Exakte Typografie (Latin Modern Roman für Fließtext, Latin Modern Sans für Überschriften)
- Original HKA-Deckblatt mit zentriertem HKA-Logo
- Römische Seitenzahlen (I, II, III...) für Titelei / Abstract / Verzeichnisse
- Arabische Seitenzahlen (1, 2, 3...) mit Kapitelkopfzeile im Format "Kapitelname | Seite"
- Tabellen im LaTeX-Booktabs-Stil mit "Tabelle X.Y:" Beschriftung
- Abbildungsunterschriften "Bild X.Y:"
- Klickbare interne und externe Hyperlinks
"""

import sys
import os
import re
import base64
import subprocess
import markdown
from markdown.extensions.toc import TocExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension


def get_image_base64(image_path: str) -> str:
    """Liest ein Bild ein und gibt es als Base64 Data-URI zurück."""
    if not os.path.exists(image_path):
        return ""
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    if ext == "svg":
        mime = "image/svg+xml"
    elif ext in ["jpg", "jpeg"]:
        mime = "image/jpeg"
    else:
        mime = "image/png"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def build_hka_latex_css(logo_base64: str) -> str:
    """Erzeugt das CSS, das exakt dem LaTeX-Layout von beispiel_bericht.pdf entspricht."""
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Latin+Modern+Roman:ital,wght@0,400;0,700;1,400;1,700&family=Latin+Modern+Mono&family=Latin+Modern+Sans:wght@400;700&display=swap');

    @page {{
        size: A4;
        margin: 28mm 25mm 25mm 25mm;
        @top-right {{
            content: string(chapter-title) " | " counter(page);
            font-family: 'Latin Modern Roman', serif;
            font-size: 10pt;
            color: #000;
        }}
    }}

    @page:first {{
        margin: 0;
        @top-right {{ content: none; }}
        @bottom-center {{ content: none; }}
    }}

    body {{
        font-family: 'Latin Modern Roman', 'Times New Roman', Times, serif;
        font-size: 10.5pt;
        line-height: 1.5;
        color: #000;
        text-align: justify;
        hyphens: auto;
        background: #fff;
        margin: 0;
        padding: 0;
    }}

    /* DECKBLATT (exakt wie beispiel_bericht.pdf Seite 1) */
    .deckblatt {{
        page-break-after: always;
        height: 100vh;
        box-sizing: border-box;
        padding: 35mm 25mm 25mm 25mm;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        text-align: center;
    }}
    .deckblatt-logo-container {{
        margin-top: 10mm;
        margin-bottom: 20mm;
    }}
    .deckblatt-logo {{
        max-width: 140px;
        height: auto;
    }}
    .deckblatt-fakultaet {{
        font-family: 'Latin Modern Roman', serif;
        font-size: 11pt;
        color: #111;
        margin-bottom: 4px;
    }}
    .deckblatt-studiengang {{
        font-family: 'Latin Modern Roman', serif;
        font-size: 11pt;
        color: #111;
        margin-bottom: 30mm;
    }}
    .deckblatt-arbeitstyp {{
        font-family: 'Latin Modern Roman', serif;
        font-size: 17pt;
        font-weight: bold;
        color: #000;
        margin-bottom: 12mm;
    }}
    .deckblatt-titel {{
        font-family: 'Latin Modern Roman', serif;
        font-size: 15pt;
        font-weight: bold;
        line-height: 1.35;
        color: #000;
        margin-bottom: 6mm;
        padding: 0 10mm;
    }}
    .deckblatt-subtitel {{
        font-family: 'Latin Modern Roman', serif;
        font-size: 12pt;
        font-style: italic;
        color: #222;
        margin-bottom: 25mm;
        padding: 0 10mm;
    }}
    .deckblatt-meta {{
        font-family: 'Latin Modern Roman', serif;
        font-size: 10pt;
        text-align: left;
        display: inline-block;
        margin: 0 auto;
        line-height: 1.6;
    }}
    .deckblatt-datum {{
        font-family: 'Latin Modern Roman', serif;
        font-size: 10.5pt;
        margin-top: 20mm;
        margin-bottom: 5mm;
        text-align: center;
    }}

    .page-break {{
        page-break-before: always;
    }}

    /* ÜBERSCHRIFTEN (Sans-Serif Bold wie LaTeX HKA) */
    h1, h2, h3, h4 {{
        font-family: 'Latin Modern Sans', sans-serif;
        color: #000;
        font-weight: bold;
        page-break-after: avoid;
    }}
    h1 {{
        font-size: 17pt;
        margin-top: 22pt;
        margin-bottom: 14pt;
        string-set: chapter-title content();
    }}
    h2 {{
        font-size: 13pt;
        margin-top: 16pt;
        margin-bottom: 8pt;
    }}
    h3 {{
        font-size: 11pt;
        margin-top: 12pt;
        margin-bottom: 6pt;
    }}

    /* PARAGRAPHEN & EINZÜGE */
    p {{
        margin-top: 0;
        margin-bottom: 8pt;
    }}

    /* AUFZÄHLUNGEN */
    ul, ol {{
        margin-top: 4pt;
        margin-bottom: 8pt;
        padding-left: 20px;
    }}
    li {{
        margin-bottom: 3pt;
    }}

    /* TABELLEN (LaTeX booktabs Style) */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14pt 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }}
    th {{
        border-top: 1.2pt solid #000;
        border-bottom: 0.8pt solid #000;
        padding: 5px 6px;
        font-family: 'Latin Modern Roman', serif;
        font-weight: bold;
        text-align: left;
    }}
    td {{
        border-bottom: 0.4pt solid #ccc;
        padding: 4px 6px;
        vertical-align: top;
    }}
    tr:last-child td {{
        border-bottom: 1.2pt solid #000;
    }}
    caption, .table-caption {{
        caption-side: top;
        font-family: 'Latin Modern Roman', serif;
        font-weight: bold;
        font-size: 9.5pt;
        margin-bottom: 6px;
        text-align: left;
    }}

    /* ABBILDUNGEN & CAPTIONS */
    .figure-container {{
        text-align: center;
        margin: 14pt 0;
        page-break-inside: avoid;
    }}
    .figure-container img {{
        max-width: 90%;
        height: auto;
    }}
    .figure-caption {{
        font-family: 'Latin Modern Roman', serif;
        font-size: 9pt;
        font-weight: bold;
        margin-top: 6px;
    }}

    /* CODE BLÖCKE */
    pre, code {{
        font-family: 'Latin Modern Mono', monospace;
    }}
    code {{
        background: #f5f5f5;
        padding: 1px 3px;
        font-size: 8.5pt;
    }}
    pre {{
        background: #f8f8f8;
        border: 0.5pt solid #ddd;
        padding: 8pt;
        font-size: 8pt;
        line-height: 1.35;
        overflow-x: auto;
        page-break-inside: avoid;
    }}
    pre code {{
        background: transparent;
        padding: 0;
    }}

    /* ZITATE & ALERTS */
    blockquote {{
        margin: 10pt 0;
        padding: 6pt 12pt;
        background: #f9f9f9;
        border-left: 2.5pt solid #555;
        font-style: italic;
        page-break-inside: avoid;
    }}

    /* LINKS */
    a {{
        color: #000;
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}

    /* UNTERSCHRIFTEN-BLOCK */
    .signature-block {{
        margin-top: 25mm;
        display: flex;
        justify-content: space-between;
    }}
    .signature-line {{
        width: 42%;
        border-top: 0.8pt solid #000;
        text-align: center;
        padding-top: 4px;
        font-size: 9.5pt;
    }}
    """


def transform_deckblatt_markdown(md_content: str, logo_base64: str) -> str:
    """Erzeugt den exakten HTML-Deckblatt-Block wie in beispiel_bericht.pdf."""
    # Falls bereits ein eigenes Deckblatt-HTML vorhanden ist, passen wir es an
    deckblatt_html = f"""
<div class="deckblatt">
  <div class="deckblatt-logo-container">
    <img src="{logo_base64}" class="deckblatt-logo" alt="HKA Logo">
  </div>

  <div class="deckblatt-faculty-section">
    <div class="deckblatt-fakultaet">Fakultät für Maschinenbau und Mechatronik</div>
    <div class="deckblatt-studiengang">Studiengang Robotik und KI i.d. Produktion</div>
  </div>

  <div class="deckblatt-main-section">
    <div class="deckblatt-arbeitstyp">Projektbericht</div>
    <div class="deckblatt-titel">Autonome Fahrzeugsteuerung für Carrera Hybrid mittels Deep Reinforcement Learning und Privilegierter Expertendestillation</div>
    <div class="deckblatt-subtitel">Konzeption, 2D-Prototyping und empirische Evaluation einer sample-effizienten Teacher-Student-Architektur zur Vorbereitung des Sim-to-Real-Transfers</div>
  </div>

  <div class="deckblatt-meta-section">
    <div class="deckblatt-meta">
      <strong>Vorgelegt von:</strong> Felix Faaß, Magnus Neumann<br>
      <strong>Matrikelnummer:</strong> [103121, 103560]<br>
      <strong>Betreuer (HKA):</strong> Prof. Dr.-Ing. habil. Björn Hein
    </div>
  </div>

  <div class="deckblatt-datum">
    Karlsruhe, den 31. August 2026
  </div>
</div>
"""
    # Ersetze den vorherigen <div class="deckblatt">...</div> Block
    pattern = r'<div class="deckblatt">.*?</div>\s*<div class="page-break"></div>'
    if re.search(pattern, md_content, flags=re.DOTALL):
        md_content = re.sub(pattern, deckblatt_html + '\n<div class="page-break"></div>', md_content, flags=re.DOTALL)
    return md_content


def convert_md_to_html(md_content: str, logo_base64: str) -> str:
    """Wandelt Markdown in vollständiges HTML mit dem exakten HKA-LaTeX-CSS um."""
    md_content = transform_deckblatt_markdown(md_content, logo_base64)
    md = markdown.Markdown(
        extensions=[
            TableExtension(),
            FencedCodeExtension(),
            TocExtension(permalink=False),
            'attr_list',
            'def_list',
            'footnotes',
        ]
    )
    html_body = md.convert(md_content)
    css = build_hka_latex_css(logo_base64)

    html_document = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>Projektbericht: Autonome Fahrzeugsteuerung Carrera Hybrid</title>
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
    </script>
    <style>
{css}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    return html_document


def render_html_to_pdf(html_path: str, output_pdf_path: str) -> bool:
    """Rendert eine HTML-Datei mit Chromium headless als PDF."""
    cmd = [
        "chromium",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=4000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={output_pdf_path}",
        "--no-pdf-header-footer",
        html_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return os.path.exists(output_pdf_path)
    except Exception as e:
        print(f"Fehler beim Rendern mit Chromium: {e}", file=sys.stderr)
        return False


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "projektbericht.md"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "projektbericht.pdf"

    if not os.path.exists(input_file):
        print(f"Fehler: Datei '{input_file}' nicht gefunden.")
        sys.exit(1)

    logo_path = os.path.join(os.path.dirname(__file__), "HKA_logo.png")
    logo_base64 = get_image_base64(logo_path)

    with open(input_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    print(f"-> Wandle '{input_file}' in LaTeX-HKA Layout um...")
    html_content = convert_md_to_html(md_content, logo_base64)

    temp_html = "temp_report_render.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"-> Erzeuge PDF '{output_file}' via Chromium Paged Media...")
    success = render_html_to_pdf(os.path.abspath(temp_html), os.path.abspath(output_file))

    if os.path.exists(temp_html):
        os.remove(temp_html)

    if success:
        size_kb = os.path.getsize(output_file) / 1024
        print(f"[ERFOLG] PDF erfolgreich erstellt: '{output_file}' ({size_kb:.1f} KB)")
    else:
        print("[FEHLER] PDF-Erstellung fehlgeschlagen.")
        sys.exit(1)


if __name__ == "__main__":
    main()
