#!/bin/bash

NUM_PROCS=24
TARGET=config.mk
EXP_DIRS=experiment/exp-*

make -r build_ext
find $EXP_DIRS -type f -name $TARGET -print0 | \
    while IFS= read -r -d '' config_file; do
        experiment=`dirname "$config_file"`
        echo "replicating '$experiment'"
        touch "$experiment/$TARGET"
        EXPERIMENT="$experiment" make -r -j$NUM_PROCS analysis
    done
