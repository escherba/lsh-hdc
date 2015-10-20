#!/bin/bash

NUM_PROCS=24

make -r build_ext
ls -d experiment/out-*/config.mk | \
    while read f; do
        dname=`dirname $f`
        echo "replicating $dname"
        touch $f
        OUTPUT_DIR=$dname make -r -j$NUM_PROCS analysis
    done
