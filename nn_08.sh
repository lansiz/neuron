#!/bin/sh

nums="0 1 2 3 4 5 6 7 8 9"
for num in $nums; do
    python nn_08_too_many_fp.py &
done

