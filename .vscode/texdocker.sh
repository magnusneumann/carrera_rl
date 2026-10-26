#!/bin/sh
# Fuehrt ein TeX-Kommando (pdflatex, biber, makeindex, ...) im TeX-Live-Container aus.
# Arbeitsverzeichnis = aktuelles Verzeichnis, Dateien gehoeren dem aufrufenden Benutzer.
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$(pwd):$(pwd)" -w "$(pwd)" texlive/texlive:latest "$@"
