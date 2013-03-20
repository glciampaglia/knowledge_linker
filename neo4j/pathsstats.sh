#!/bin/bash

STATEMENTS=statements.txt
SCRIPT=paths.py
MAXLEN=3
PREF=http://dbpedia.org/resource/

exec 3<$STATEMENTS

while `true` ; do
    read -u 3 source target
    if [[ $? != 0 ]] ; then
        break
    fi
    echo "## ${source##$PREF} -> ${target##$PREF}"
    echo "## ${source##$PREF} -> ${target##$PREF}" 1>&2
    python $SCRIPT -l $MAXLEN --debug -s $source -t $target 
done
