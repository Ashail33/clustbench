# Build both papers. Requires a TeX distribution (pdflatex + bibtex) and,
# for `make figures`, matplotlib.
PAPERS = clustbench/clustbench.pdf algorithms/algorithms.pdf

all: $(PAPERS)

figures:
	python scripts/make_figures.py

%.pdf: %.tex references.bib
	cd $(dir $<) && pdflatex -interaction=nonstopmode $(notdir $<) >/dev/null \
	  && bibtex $(basename $(notdir $<)) >/dev/null \
	  && pdflatex -interaction=nonstopmode $(notdir $<) >/dev/null \
	  && pdflatex -interaction=nonstopmode $(notdir $<) >/dev/null

clean:
	rm -f */*.aux */*.bbl */*.blg */*.log */*.out

.PHONY: all figures clean
