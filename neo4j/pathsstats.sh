#!/bin/bash

STATEMENTS=statements.txt
SCRIPT=paths.py
MAXLEN=20
PREF=http://dbpedia.org/resource/

exec 3<$STATEMENTS

while `true` ; do
    read -u 3 source target
    if [[ $? != 0 ]] ; then
        break
    fi
    echo "# ---------------------------------------------------" | tee /dev/stderr
    echo "# ${source##$PREF} -> ${target##$PREF}" | tee /dev/stderr
    echo "# ---------------------------------------------------" | tee /dev/stderr
    python $SCRIPT -l $MAXLEN --debug -s $source -t $target 
done
