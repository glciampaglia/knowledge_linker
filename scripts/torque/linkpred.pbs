#!/bin/bash
#PBS -l walltime=12:00:00,nodes=1:ppn=8,mem=8G
#PBS -V
#PBS -j oe
#PBS -d /N/u/gciampag/BigRed2/truthy_measure/experiments/2013-09-12-statements
#PBS -N linkpred
#PBS -t 0-69

#   Copyright 2016 The Trustees of Indiana University.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# walltime for 50 lines: 2h for directed, 12h for undirected
# walltime per line (w/o paths): 15' undirected,  13s directed

SCRIPT=klinker linkpred.
ROOT=${HOME}/data/dbpedia
ADJ=${ROOT}/filtered/adjacency.npy
NODES=${ROOT}/filtered/nodes_uris.txt

shopt -s failglob
inputs=(${ROOT}/relation_extraction/{birthplace,deathplace,degree,institution}??)

metric=metric
dir=
weight=logdegree

echo "`date`: Job ${PBS_JOBID:=TEST} index ${PBS_ARRAYID:=0} started on ${PBS_SERVER:=`hostname`}."

input=${inputs[${PBS_ARRAYID}]}
base=$(basename ${input})
name=${base%.txt}
output=${name}-${metric}${dir}_${PBS_JOBID/\[[0-9]*\]/}.json
error=${name}-${metric}${dir}_${PBS_JOBID/\[[0-9]*\]/}.err
set -x
${SCRIPT} -n 8 ${dir} -k ${metric} -w ${weight} ${NODES} ${ADJ} ${input} ${output} 2>${error}
set +x

echo All tasks terminated.

echo "`date`: Job ${PBS_JOBID} index ${PBS_ARRAYID} terminated on ${PBS_SERVER}."

# vim: sts=4 sw=4 expandtab nowrap:
