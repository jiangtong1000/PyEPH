#!/bin/bash
# Collect a completed QE phonon calculation into the layout expected by qe2pert.x.
#
# Required configuration:
#   PREFIX=my_material bash ph_collect.sh
#
# Optional environment variables: WORK_ROOT, TMP_ROOT, SAVE_DIR, and DYN0.
# The script writes to SAVE_DIR.partial and refuses existing outputs.

set -euo pipefail

PREFIX="${PREFIX:-PREFIX}"
WORK_ROOT="${WORK_ROOT:-.}"
TMP_ROOT="${TMP_ROOT:-${WORK_ROOT}/tmp}"
SAVE_DIR="${SAVE_DIR:-${WORK_ROOT}/save}"
DYN0="${DYN0:-${WORK_ROOT}/${PREFIX}.dyn0}"
STAGE_DIR="${SAVE_DIR}.partial"
PHSAVE_SOURCE="${TMP_ROOT}/_ph0/${PREFIX}.phsave"

fail() {
  echo "error: $*" >&2
  exit 2
}

require_file() {
  [ -f "$1" ] || fail "missing required file: $1"
}

if [ "$PREFIX" = "PREFIX" ]; then
  fail "set PREFIX to the calculation prefix"
fi
[ -d "$TMP_ROOT" ] || fail "missing scratch directory: $TMP_ROOT"
[ -d "$PHSAVE_SOURCE" ] || fail "missing consolidated phsave: $PHSAVE_SOURCE"
require_file "$DYN0"
[ ! -e "$SAVE_DIR" ] || fail "refusing existing output: $SAVE_DIR"
[ ! -e "$STAGE_DIR" ] || fail "refusing existing staging output: $STAGE_DIR"

NQ=$(awk 'NR == 2 {print $1; exit}' "$DYN0")
case "$NQ" in
  ''|*[!0-9]*) fail "cannot read the q-point count from line 2 of $DYN0" ;;
esac
[ "$NQ" -ge 1 ] || fail "q-point count must be positive"

SELECTED_SOURCE=""
select_identical_source() {
  local label=$1
  shift
  local candidates=()
  local path
  for path in "$@"; do
    if [ -s "$path" ]; then
      candidates+=("$path")
    fi
  done
  [ "${#candidates[@]}" -gt 0 ] || fail "no nonempty source for $label"

  local reference=${candidates[0]}
  local reference_size reference_hash
  reference_size=$(stat -c %s "$reference")
  reference_hash=""
  if [ "${#candidates[@]}" -gt 1 ]; then
    reference_hash=$(sha256sum "$reference" | awk '{print $1}')
    for path in "${candidates[@]:1}"; do
      [ "$(stat -c %s "$path")" = "$reference_size" ] || \
        fail "conflicting sizes for $label: $reference and $path"
      [ "$(sha256sum "$path" | awk '{print $1}')" = "$reference_hash" ] || \
        fail "conflicting contents for $label: $reference and $path"
    done
  fi
  SELECTED_SOURCE=$reference
}

mkdir -p "$STAGE_DIR/${PREFIX}.phsave"
cp -a "$PHSAVE_SOURCE/." "$STAGE_DIR/${PREFIX}.phsave/"
cp "$DYN0" "$STAGE_DIR/${PREFIX}.dyn0"

dvscf_size=""
for q_index in $(seq 1 "$NQ"); do
  require_file "$PHSAVE_SOURCE/patterns.${q_index}.xml"
  require_file "$PHSAVE_SOURCE/dynmat.${q_index}.0.xml"

  dyn_candidates=(
    "${WORK_ROOT}/${PREFIX}.dyn${q_index}.xml"
  )
  select_identical_source "dynamical matrix q${q_index}" "${dyn_candidates[@]}"
  cp "$SELECTED_SOURCE" "$STAGE_DIR/${PREFIX}.dyn${q_index}.xml"

  dvscf_candidates=()
  if [ "$q_index" -eq 1 ]; then
    dvscf_candidates+=("${TMP_ROOT}/_ph0/${PREFIX}.dvscf1")
    dvscf_candidates+=("${TMP_ROOT}/_ph0/${PREFIX}.q_1/${PREFIX}.dvscf1")
  else
    while IFS= read -r -d '' path; do
      dvscf_candidates+=("$path")
    done < <(
      find "$TMP_ROOT" -type f \
        -path "*/${PREFIX}.q_${q_index}/${PREFIX}.dvscf1" -print0
    )
  fi
  select_identical_source "dvscf q${q_index}" "${dvscf_candidates[@]}"

  current_size=$(stat -c %s "$SELECTED_SOURCE")
  if [ -z "$dvscf_size" ]; then
    dvscf_size=$current_size
  elif [ "$current_size" != "$dvscf_size" ]; then
    fail "dvscf q${q_index} has $current_size bytes; expected $dvscf_size"
  fi
  cp "$SELECTED_SOURCE" "$STAGE_DIR/${PREFIX}.dvscf_q${q_index}"
  [ "$(stat -c %s "$STAGE_DIR/${PREFIX}.dvscf_q${q_index}")" = "$current_size" ] || \
    fail "short destination copy for dvscf q${q_index}"
  echo "q${q_index}: $SELECTED_SOURCE"
done

mv "$STAGE_DIR" "$SAVE_DIR"
echo "collected $NQ q points with $dvscf_size-byte dvscf files into $SAVE_DIR"
