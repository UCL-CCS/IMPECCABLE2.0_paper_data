#!/bin/bash

let "ndir = $1 - 1"

eval cat mmpbsa/{0..$ndir}/_MMPBSA_complex_pb.mdout.0 | awk '($1=="BOND"||$1=="VDWAALS"||$1=="1-4")' | cut -c11-24,36-49,64-77|paste - - -|awk '{s=0;for(i=1;i<=NF;i++){s+=$i}; print s"\t"NR}' > mmpbsa-com.dat
eval cat mmpbsa/{0..$ndir}/_MMPBSA_receptor_pb.mdout.0 | awk '($1=="BOND"||$1=="VDWAALS"||$1=="1-4")' | cut -c11-24,36-49,64-77|paste - - -|awk '{s=0;for(i=1;i<=NF;i++){s+=$i}; print s"\t"NR}' > mmpbsa-rec.dat
eval cat mmpbsa/{0..$ndir}/_MMPBSA_ligand_pb.mdout.0 | awk '($1=="BOND"||$1=="VDWAALS"||$1=="1-4")' | cut -c11-24,36-49,64-77|paste - - -|awk '{s=0;for(i=1;i<=NF;i++){s+=$i}; print s"\t"NR}' > mmpbsa-lig.dat

if [ ! -s mmpbsa-com.dat ] || [ $(wc -l <mmpbsa-com.dat) -ne $(wc -l <mmpbsa-rec.dat) ] || [ $(wc -l <mmpbsa-com.dat) -ne $(wc -l <mmpbsa-lig.dat) ]; then
	echo -e "NA\tNA" > dg_poses.dat
else
	paste mmpbsa-com.dat mmpbsa-rec.dat mmpbsa-lig.dat | awk '{print $1-$3-$5"\t"$6}' > dg_poses.dat
fi

#sort -n dg.dat | head -1|awk -v l=$l '{print l"\t"$2}' >> lig-pose.dat

# > lig-pose.dat
