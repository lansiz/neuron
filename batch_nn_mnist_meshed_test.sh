#!/bin/sh

nums="0 1 2 3 4 5 6 7 8 9"
echo '' > nn_mnist_meshed_test_${1}.log
for train_num in $nums; do
    for test_num in $nums; do
        python nn_mnist_meshed_test.py -p nn_mnist_strength_${train_num}_${1}.pkl -i 1000 -m $train_num -n $test_num  >> nn_mnist_meshed_test_${1}.log &
    done
done

