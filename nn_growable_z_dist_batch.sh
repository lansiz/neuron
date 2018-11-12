#!/bin/sh
nums="0 1 2 3 4 5 6 7 8 9"
for test_num in $nums; do
python nn_growable_dist.py -j $test_num &
done

