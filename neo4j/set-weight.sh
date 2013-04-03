#!/bin/bash

seq -f '%1.f' 700000 6875246 | xargs -n 10000 -P10 ./runquery.sh
