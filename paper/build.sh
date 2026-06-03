#!/usr/bin/env bash
# Build paper/main.tex into main.pdf with bibtex, then sanity-check the log.
#
# Usage:
#   ./build.sh           # full build (pdflatex -> bibtex -> pdflatex x3)
#   ./build.sh --quick   # single pdflatex pass (no bibtex), for fast text-only edits
#   ./build.sh --clean   # remove build artifacts, then full build
#
# Runs from anywhere: it cd's into its own directory first.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

DOC=main
PDFLATEX="pdflatex -interaction=nonstopmode -halt-on-error"
LOG=/tmp/${DOC}_build.log

clean() {
    rm -f "${DOC}".{aux,bbl,blg,out,log,toc,lof,lot,fls,fdb_latexmk}
}

case "${1:-}" in
    --clean) clean ;;
    --quick)
        echo ">> quick build (1 pass, no bibtex)"
        $PDFLATEX "$DOC".tex > "$LOG" 2>&1 || { tail -30 "$LOG"; exit 1; }
        QUICK=1 ;;
esac

if [[ "${QUICK:-0}" != "1" ]]; then
    echo ">> pass 1/4 (pdflatex)"
    $PDFLATEX "$DOC".tex > "$LOG" 2>&1 || { tail -30 "$LOG"; exit 1; }
    echo ">> bibtex"
    bibtex "$DOC" > /tmp/${DOC}_bibtex.log 2>&1 || { cat /tmp/${DOC}_bibtex.log; exit 1; }
    echo ">> pass 2/4 (pdflatex)"
    $PDFLATEX "$DOC".tex > "$LOG" 2>&1 || { tail -30 "$LOG"; exit 1; }
    echo ">> pass 3/4 (pdflatex)"
    $PDFLATEX "$DOC".tex > "$LOG" 2>&1 || { tail -30 "$LOG"; exit 1; }
    echo ">> pass 4/4 (pdflatex)"
    $PDFLATEX "$DOC".tex > "$LOG" 2>&1 || { tail -30 "$LOG"; exit 1; }
fi

echo
grep "Output written" "$LOG" || true
echo
echo "==== sanity check ===="
issues=0
report() { # label, grep-pattern
    local hits
    hits=$(grep -icE "$2" "$LOG" || true)
    if [[ "$hits" -gt 0 ]]; then
        printf '  [!] %-22s %s hit(s)\n' "$1" "$hits"
        issues=$((issues + hits))
    else
        printf '  [ok] %-21s none\n' "$1"
    fi
}
report "undefined references" "undefined (references|There were undefined references)"
report "undefined citations"  "undefined citations|Citation .* undefined"
report "rerun needed"         "Rerun to get|Label\(s\) may have changed"
report "overfull hbox"        "Overfull \\\\hbox"

echo
if [[ "$issues" -eq 0 ]]; then
    echo "All clean. -> $(pwd)/${DOC}.pdf"
else
    echo "$issues issue(s) flagged. Full log: $LOG"
    exit 1
fi
