#!/bin/bash
# Helper script to create symlinks so that each step can find
# the tmp/ directory from the SCF calculation.
#
# Run this from the project root directory AFTER step 1 (SCF) completes.
# Steps 2-6 all need access to the SCF tmp/ directory.
#
# Usage:  cd /path/to/my_project && bash setup_links.sh

PREFIX='PREFIX'   # TODO: same prefix

SCF_DIR="$(pwd)/1_SCF"

for step in 2_D3HESS 3_PHONONS 4_NSCF 5_WANN 6_QE2PERT; do
    if [ -d "$step" ]; then
        echo "Linking $step/tmp -> $SCF_DIR/tmp"
        ln -sfn "$SCF_DIR/tmp" "$step/tmp"
    fi
done

# Step 3 also needs the D3 hessian file (if using grimme-d3)
if [ -d "2_D3HESS" ] && [ -d "3_PHONONS" ]; then
    echo "Linking 3_PHONONS/${PREFIX}.hess -> 2_D3HESS/${PREFIX}.hess"
    ln -sfn "$(pwd)/2_D3HESS/${PREFIX}.hess" "3_PHONONS/${PREFIX}.hess"
fi

echo "Done! Symlinks created."
