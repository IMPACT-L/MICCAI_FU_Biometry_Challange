# MWM 2026 Overleaf Package

This folder is an Overleaf-ready Springer LNCS paper package for the MWM 2026 workshop.

## Requirements used

From the official workshop site:

- max `8 pages + 2 references`
- `Springer LNCS` format
- `single-blind` review
- OpenReview submission

Source:

- <https://mwm2026.github.io/>

## Files

- `main.tex`: main manuscript
- `references.bib`: BibTeX database
- `llncs.cls`: Springer LNCS class file
- `splncs04.bst`: Springer bibliography style
- `assets/`: figures already referenced in the paper

## Before submission

Replace these placeholders in `main.tex`:

- author names
- affiliations
- contact email
- acknowledgments
- any result values you want to update

## Overleaf

1. Create a new blank project on Overleaf.
2. Upload the full contents of this folder.
3. Set `main.tex` as the main file if Overleaf does not detect it automatically.
4. Compile with `pdfLaTeX`.

## Local compile

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
