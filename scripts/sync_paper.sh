#!/bin/bash
# Copy the generated figures and tables into paper/, so the paper folder is self-contained
# (upload paper/ to Overleaf as it is). Rerun after any change to the results.
set -e
cd "$(dirname "$0")/.."
mkdir -p paper/figures paper/tables
cp results/figures/*.pdf paper/figures/
cp results/tables/*.tex paper/tables/
echo "copied $(ls results/figures/*.pdf | wc -l) figures and $(ls results/tables/*.tex | wc -l) tables into paper/"
