#!/bin/bash

# Collect phonon data into save/ directory for qe2pert.x
#
# IMPORTANT: When ph.x is run with -ni N (image parallelization),
# the dvscf files for q-point i are stored in tmp/_ph{i-1}/

PREFIX='PREFIX' # TODO: same prefix as in ph.in

#ph-collect.sh should be in the work directory of PHonon calculation

echo `date`
echo `pwd`

echo 'PREFIX: ' $PREFIX
echo "Creating a save dir..."
mkdir -p save/${PREFIX}.phsave

PH0_DIR="tmp/_ph0"

echo "Copying prefix.phsave..."
cp ${PH0_DIR}/${PREFIX}.phsave/* save/${PREFIX}.phsave/

echo "Copying dyn files..."
cp ./${PREFIX}.dyn* save/

echo "Copying the dvscf file for the first q-point..."
cp ${PH0_DIR}/${PREFIX}.dvscf1 save/${PREFIX}.dvscf_q1

echo "Copy the dvscf for q-points > 1..."
for q_folder in ${PH0_DIR}/${PREFIX}.q_*; do
   echo $q_folder;
   NQ=`echo $q_folder | awk -F_ '{print $NF}'`;
   # cp ${PH0_DIR}/${PREFIX}.q_${NQ}/${PREFIX}.dvscf1 save/${PREFIX}.dvscf_q${NQ} # original
   cp tmp/_ph$((NQ-1))/${PREFIX}.q_${NQ}/${PREFIX}.dvscf1 save/${PREFIX}.dvscf_q${NQ} # my fix here
done

echo "Done!"
echo `date`