#!/bin/sh
echo '' > nn_meshed_test_number.log
nums="0 1 2 3 4 5 6 7 8 9"
for test_num in $nums; do
python nn_meshed_classification.py -i 10000 -j $test_num >> nn_meshed_test_number.log &
done

