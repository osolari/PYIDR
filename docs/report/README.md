# sAIm LaTeX Template

A cleaned single-paper LaTeX template based on the uploaded project.

## Files
- `main.tex`: main entry point
- `saim.cls`: journal-style class file with the sAIm title box
- `math_commands.tex`: common math macros
- `sections/`: modular section files
- `refs.bib`: sample bibliography
- `tables/`: example standalone table snippets

## Build
```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Notes
- Replace the placeholder text in `sections/` with your manuscript.
- Update authors, affiliations, metadata, and bibliography in `main.tex`.
- The title box uses `assets/saim_logo.png`.
