#!/bin/bash

NUM_PROCS=24

make -r build_ext
find experiment/exp-* -type f -name config.mk -print0 | \
    while IFS= read -r -d '' config_file; do
        experiment=`dirname "$config_file"`
        echo "replicating '$experiment'"
        touch "$config_file"
        EXPERIMENT="$experiment" make -r -j$NUM_PROCS analysis
    done
