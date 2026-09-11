# Makefile für LaTeX-thesis mit BibLaTeX und nomencl, ohne latexmk

MAIN  = main
PDF   = $(MAIN).pdf
BIB   = bibliography.bib

LATEX = pdflatex -interaction=nonstopmode

# Plattformabhängiger Löschbefehl
ifeq ($(OS),Windows_NT)
  # Windows: nutze cmd.exe-del
  RM = cmd /C del /Q /F
else
  # Unix/Mac: normales rm
  RM = rm -f
endif

# Standard: Voller Build (alles)
all: full

# Alias für Kompatibilität
compile: full

# Vollständiger Build: pdflatex + biber + nomencl + pdflatex x2
full:
	@echo "Vollständiger Build (pdflatex + biber + nomencl + pdflatex x2)..."
	$(LATEX) $(MAIN)
	biber $(MAIN)
	makeindex $(MAIN).nlo -s nomencl.ist -o $(MAIN).nls
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Schneller Build: nur einmal pdflatex
fast:
	@echo "Schneller Build (einmal pdflatex)..."
	$(LATEX) $(MAIN)

# Alias für fast
pdflatex: fast

# Nur Literatur aktualisieren (mit pdflatex davor und danach)
biber:
	@echo "Biber-Lauf mit pdflatex davor und danach..."
	$(LATEX) $(MAIN)
	biber $(MAIN)
	$(LATEX) $(MAIN)

# Nur Nomenklatur aktualisieren (mit pdflatex davor und danach)
nomen:
	@echo "Nomenklatur-Lauf (pdflatex + makeindex + pdflatex)..."
	$(LATEX) $(MAIN)
	makeindex $(MAIN).nlo -s nomencl.ist -o $(MAIN).nls
	$(LATEX) $(MAIN)

# Aufräumen der Hilfsdateien
clean:
	-$(RM) *.aux *.log *.toc *.lof *.lot *.bbl *.blg *.out *.fls *.fdb_latexmk \
	       *.nav *.snm *.vrb *.thm *.bcf *.run.xml *.nlo *.nls *.ilg *.aux
	@echo "Cleaned auxiliary files."

# Alles inkl. PDF entfernen
fclean: clean
	-$(RM) $(PDF)

.PHONY: all full fast pdflatex biber nomen clean fclean compile