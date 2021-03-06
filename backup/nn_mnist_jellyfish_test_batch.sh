#!/bin/sh
log='nn_mnist_jellyfish_test.log'
nums="0 1 2 3 4 5 6 7 8 9"
echo '' > $log
for train_num in $nums; do
    for test_num in $nums; do
        python nn_mnist_jellyfish_test.py -p nn_mnist_jellyfish_${train_num}.pkl -i 10000 -m $train_num -n $test_num  >> $log 2>>log &
    done
done

