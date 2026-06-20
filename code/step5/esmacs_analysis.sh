#!/bin/bash

n_compound=$1

for l in `seq 0 $(( $n_compound - 1 ))`; do
	for d in `ls -d $l/replicas/rep*/simulation`; do
#		cat $d/*/FINAL_RESULTS_MMPBSA.dat | grep "DELTA TOTAL" | awk '(NR%2==0) {print $3}'
		for f in `ls $d/*/FINAL_RESULTS_MMPBSA.dat`; do
			tail -20 $f |awk '($1=="VDWAALS"||$1=="EEL"||$1=="EPB"||$1=="ENPOLAR") {s+=$2} END {print s}'
		done | awk -f $2/ave.awk
	done | awk -f $2/ave.awk 
done > dg.dat

paste $3/fixpka_compounds.smi dg.dat | awk '{print $1","$3","$4}' > smiles-dg.csv

# sed -i "1i OE_SMILES,dG_ESMACS,sd_ESMACS" smiles-dg.csv

