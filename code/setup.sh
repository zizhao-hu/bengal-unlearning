#!/bin/bash
# Build an open-unlearning checkout that can run SHRED on Hubble.
#   bash setup.sh [target_dir]      (default: ./open-unlearning)
# Clones upstream open-unlearning at the commit our runs used (a456aa2),
# applies the small registry patch, and copies in the SHRED trainer, the
# Hubble configs/evaluator/data and the run scripts.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-open-unlearning}"
UPSTREAM_COMMIT=a456aa2

if [[ ! -d "$DEST/.git" ]]; then
    git clone https://github.com/locuslab/open-unlearning.git "$DEST"
fi
cd "$DEST"
git checkout -q "$UPSTREAM_COMMIT"
git apply "$HERE/patches/open-unlearning-a456aa2.patch"
cp -R "$HERE/overlay/." .
chmod +x scripts/hubble/*.sh
echo "Ready: $(pwd)"
echo "Next: install open-unlearning's requirements (see its README), then run scripts/hubble/run_hubble.sh"
