#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$PYTHONPATH
echo $PYTHONPATH

pytest
