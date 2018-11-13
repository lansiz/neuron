#!/bin/sh

echo '' > ./nn_meshed.log
nums="0 1 2 3 4 5 6 7 8 9"
for num in $nums; do
python nn_meshed_train.py -m $num >> ./nn_meshed.log &
done

