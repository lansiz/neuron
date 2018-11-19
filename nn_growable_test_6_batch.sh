#!/bin/sh
echo '' > nn_growable_test_6.log
nums="0 1 2 3 4 5 6 7 8 9 10 11 12"
for test_num in $nums; do
# echo -n $test_num:
python nn_growable_classification.py -j 6 >> nn_growable_test_6.log &
done

