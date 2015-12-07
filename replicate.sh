#!/bin/bash

NUM_PROCS=`python -c "import multiprocessing as m; print m.cpu_count()"`
TARGET=config.mk
TARGET_DIRS=study-predictions/exp-*

touch *requirements.txt
make -r extras
make -r build_ext

IDX=1
TOTAL=`find $TARGET_DIRS -type f -name $TARGET | wc -l | sed 's/ *//g'`

find $TARGET_DIRS -type f -name $TARGET | \
    while read target_path; do
        experiment=`dirname "$target_path"`
        echo "Experiment $IDX out of $TOTAL: replicating '$experiment' using $NUM_PROCS processes"
        #touch "$experiment/$TARGET"
        EXPERIMENT="$experiment" make -r -j$NUM_PROCS experiment
        IDX=$(($IDX+1))
    done
