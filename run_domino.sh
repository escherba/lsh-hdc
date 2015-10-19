#!/bin/bash

make -r build_ext
make -r -j10 analysis
