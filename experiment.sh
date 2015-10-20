#!/bin/bash

NUM_PROCS=10

make -r build_ext
make -r -j$NUM_PROCS experiment
