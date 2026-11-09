# Paper (CVPR 2027 review version)

- `main.tex`: main paper (8 pages + references + LLM usage statement).
- `supp.tex`: supplementary material, compiled separately.
- `sec/`: one file per section; `fig/teaser.tex`: Figure 1.
- `figures/`, `tables/`: generated; refresh with `bash scripts/sync_paper.sh` from the repo root
  after rerunning the analysis scripts. Never edit numbers in these files by hand.
- `refs.bib`: bibliography; every entry must be checked against the paper itself.
- `cvpr.sty`, `ieeenat_fullname.bst`: from the official CVPR 2027 author kit.

Overleaf: upload this whole folder (zip it), set the main document to `main.tex`, compiler
pdfLaTeX. `\pending{...}` marks text still waiting for results and must be gone before
submission (search for it).
