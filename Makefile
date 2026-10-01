.PHONY: all pdf sim sim-all clean

all: pdf

pdf: main.pdf

main.pdf: main.tex sections/*.tex figures/*
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

sim:
	python3 spiked_svm_sim.py

sim-all: sim
	python3 spiked_svm_gram.py
	python3 spiked_svm_inconsistency.py
	python3 spiked_svm_bias.py
	python3 spiked_svm_classifiers.py

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
