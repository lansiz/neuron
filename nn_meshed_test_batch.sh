#!/bin/sh
echo '' > nn_meshed_test.log
nums="0 1 2 3 4 5 6 7 8 9 10 11 12"
for test_num in $nums; do
python nn_meshed_classification.py -i 5000 >> nn_meshed_test.log &
done

