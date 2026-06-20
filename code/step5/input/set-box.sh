#!/bin/bash

SCRIPTS_PATH=/lustre/orion/chm155/proj-shared/impeccable_data/src/esmacs/input

xbox=`grep "O *WAT" complex.pdb | awk '{print $6}' | awk -f $SCRIPTS_PATH/max-min.awk | awk '{print $NF}'`
ybox=`grep "O *WAT" complex.pdb | awk '{print $7}' | awk -f $SCRIPTS_PATH/max-min.awk | awk '{print $NF}'`
zbox=`grep "O *WAT" complex.pdb | awk '{print $8}' | awk -f $SCRIPTS_PATH/max-min.awk | awk '{print $NF}'`
x0=`grep "O *WAT" complex.pdb | awk '{print $6}' | awk -f $SCRIPTS_PATH/max-min.awk | awk '{print ($1+$2)/2}'`
y0=`grep "O *WAT" complex.pdb | awk '{print $7}' | awk -f $SCRIPTS_PATH/max-min.awk | awk '{print ($1+$2)/2}'`
z0=`grep "O *WAT" complex.pdb | awk '{print $8}' | awk -f $SCRIPTS_PATH/max-min.awk | awk '{print ($1+$2)/2}'`

printf "cellBasisVector1\\t"; printf "%6.3f " $xbox; printf " 0.000  0.000\\\\n"
printf "cellBasisVector2\\t"; printf " 0.000 "; printf "%6.3f " $ybox; printf " 0.000\\\\n"
printf "cellBasisVector3\\t"; printf " 0.000  0.000 "; printf "%6.3f\\\\n" $zbox
printf "cellOrigin\\t\\t";printf "%6.3f %6.3f %6.3f" $x0 $y0 $z0

#printf "cellBasisVector1\t"; printf "%6.3f " $xbox; printf " 0.000  0.000\\\\\n"
#printf "cellBasisVector2\t"; printf " 0.000 "; printf "%6.3f " $ybox; printf " 0.000\\\\\n"
#printf "cellBasisVector3\t"; printf " 0.000  0.000 "; printf "%6.3f\\\\\n" $zbox
#printf "cellOrigin\t\t";printf "%6.3f %6.3f %6.3f\n" $x0 $y0 $z0

