#!/bin/bash
NEO4JSHELL=$HOME/neo4j/bin/neo4j-shell
SCRIPTFILE=./weight.cypher
SCRIPT=`< $SCRIPTFILE`
DBPATH=$HOME/neo4j/dbpedia4neo/db

$NEO4JSHELL -path $DBPATH -c "$SCRIPT"

# URI='U http://dbpedia.org/resource/Barack_Obama'
# $NEO4JSHELL -path $DBPATH -c "start m = node:vertices(value='$URI') match n-[r]->m return r.degweight;"
