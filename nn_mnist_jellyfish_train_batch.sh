#!/bin/sh

echo '' > ./jellyfish.log
nums="0 1 2 3 4 5 6 7 8 9"
for num in $nums; do
python nn_mnist_jellyfish_train.py -n $num -i 20000 >> ./jellyfish.log &
done

