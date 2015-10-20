#!/bin/bash

NUM_PROCS=24
TARGET=config.mk
TARGET_DIRS=study/exp-*

make -r build_ext
find $TARGET_DIRS -type f -name $TARGET -print0 | \
    while IFS= read -r -d '' target_path; do
        experiment=`dirname "$target_path"`
        echo "replicating '$experiment'"
        touch "$experiment/$TARGET"
        EXPERIMENT="$experiment" make -r -j$NUM_PROCS experiment
    done
