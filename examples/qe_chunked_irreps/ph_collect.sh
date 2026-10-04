#!/bin/sh
# Compatibility entrypoint: portable Python owns discovery, validation and copy.
set -eu
PREFIX="${PREFIX:-PREFIX}"
WORK_ROOT="${WORK_ROOT:-.}"
exec "${PYTHON:-python}" -m pyeph.preprocessing.qe.collect_phonons \
  --prefix "$PREFIX" --work-root "$WORK_ROOT" \
  --tmp-root "${TMP_ROOT:-${WORK_ROOT}/tmp}" \
  --output "${SAVE_DIR:-${WORK_ROOT}/save}" \
  --dyn0 "${DYN0:-${WORK_ROOT}/${PREFIX}.dyn0}"
