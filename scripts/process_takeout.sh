#!/usr/bin/env bash
# Unzip any Takeout zips not yet extracted, then run every pipeline stage on the new photos.
# Incremental and safe to re-run: unzip -n never overwrites, and each stage skips finished photos.
# Runs niced (and ioniced for disk work). Log: work/process.log when started as below.
#
#   setsid nohup scripts/process_takeout.sh > work/process.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="/mnt/f/photo takeout"
DEST="$SRC/extracted"
DONE=work/unzipped.txt          # zips already extracted (so re-runs skip them quickly)
touch "$DONE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for z in "$SRC"/takeout-*.zip; do
  b=$(basename "$z")
  grep -qxF "$b" "$DONE" && continue
  echo "$(date '+%F %T') unzip $b"
  if nice -n 19 ionice -c3 unzip -n -q "$z" -d "$DEST"; then
    echo "$b" >> "$DONE"
  else
    echo "$(date '+%F %T') unzip FAILED: $b (skipped; re-run to retry)"
  fi
done

PY=(nice -n 19 .venv/bin/python -m photosel)
for stage in "ingest --workers 12" "embed" "detect" "structure" "rank" "profile office"; do
  echo "$(date '+%F %T') == $stage"
  # shellcheck disable=SC2086
  "${PY[@]}" $stage
done
echo "$(date '+%F %T') all done"
"${PY[@]}" stats
