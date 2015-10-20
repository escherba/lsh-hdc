#!/bin/bash

NUM_PROCS=16

make -r build_ext
ls -d experiment/out-*/config.mk | \
    while read f; do
        dname=`dirname $f`
        echo "Replicating $dname"
        touch $f
        OUTPUT_DIR=$dname make -r -j$NUM_PROCS analysis
    done
