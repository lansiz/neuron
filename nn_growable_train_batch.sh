#!/bin/sh

echo '' > ./nn_growable.log
nums="0 1 2 3 4 5 6 7 8 9"
for num in $nums; do
python nn_growable_train.py -n $num -i 80000 >> ./nn_growable.log &
done

