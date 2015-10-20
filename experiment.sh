#!/bin/bash

NUM_PROCS=16

make -r extras
make -r build_ext
make -r -j$NUM_PROCS experiment
