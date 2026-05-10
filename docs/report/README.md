# PY-IDR LaTeX Project

This repository contains the LaTeX source for the PY-IDR manuscript: a hierarchical nonparametric Bayesian extension of IDR via Pitman--Yor copula mixtures.

## Files

- `main.tex`: root LaTeX entry point
- `saim.cls`: project class file and title-box styling
- `math_commands.tex`: shared mathematical macros and theorem environments
- `sections/`: manuscript sections
- `tables/`: standalone table snippets included by the sections
- `figures/`: illustrative placeholder PDFs and the script used to regenerate them
- `refs.bib`: BibTeX bibliography
- `CHANGELOG.md`: applied edits from the current technical-development pass
- `CODING_AGENT_HANDOFF.md`: implementation and experiment handoff for the next iteration

## Build

The intended build uses pdfLaTeX and BibTeX:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

A `latexmkrc` file is included for Overleaf/latexmk workflows. In this container, the system `bibtex` symlink was broken, so the verified local build used `/usr/bin/bibtex.original` explicitly.

## Figure regeneration

The placeholder figures are synthetic and illustrative. They can be regenerated from the project root with:

```bash
python figures/generate_figures.py
```

The script writes output PDFs to the project-local `figures/` directory. Generated figure values are placeholders and must not be interpreted as observed experimental results.

## Manuscript status

This is a publication-development manuscript. Planned experiments, placeholder tables, illustrative figures, projected/expected behavior, and implementation tasks are intentionally retained. They should be replaced or updated only after the companion implementation and experiments are completed.
