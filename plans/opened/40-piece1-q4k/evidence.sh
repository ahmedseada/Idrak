#!/usr/bin/env bash
# evidence.sh: copies the piece's text outputs into the worktree's plans/opened/40-piece1-q4k/ (no traces, reports,
# databases, cubins or PTX files: the .nettrace, .speedscope.json, .nsys-rep, .sqlite and ptx/ stay in test-output).
src=D:/Projects/idrak/test-output/adad/piece1-q4k
dst=D:/Projects/idrak/test-output/worktrees/adad/plans/opened/40-piece1-q4k
mkdir -p "$dst/reference"
cp "$src"/*.sh "$src"/*.py "$src"/reference-*.txt "$src"/tests-*.txt "$dst/"
cp "$src"/reference/Program.cs "$src"/reference/RefHarness.csproj "$dst/reference/"
cp "$src"/reference-tuning/*.txt "$dst/reference/tuning-measured.txt"
for c in "$src"/cycle* "$src"/fix*; do
  n=$(basename "$c"); mkdir -p "$dst/$n"
  for f in "$c"/*.txt "$c"/*.csv; do [ -f "$f" ] && [ "$(basename "$f")" != bench-stdout.txt ] && cp "$f" "$dst/$n/"; done
  for d in bench-method-1 trace-run; do [ -d "$c/$d" ] && mkdir -p "$dst/$n/$d" && cp "$c/$d"/*.txt "$dst/$n/$d/"; done
  [ -d "$c/ptx" ] && mkdir -p "$dst/$n/ptxas" && cp "$c"/ptx/*.ptxas.txt "$dst/$n/ptxas/"
  [ -d "$c/probe" ] && mkdir -p "$dst/$n/probe" && cp "$c"/probe/*.ptx "$c"/probe/*.sass "$dst/$n/probe/"
done
du -sh "$dst"
