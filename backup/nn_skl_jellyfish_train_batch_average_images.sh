#!/bin/sh

echo '' > ./jellyfish.log
nums="0 1 2 3 4 5 6 7 8 9"
for num in $nums; do
python nn_skl_jellyfish_train_average_images.py -n $num -i 50000 >> ./jellyfish.log &
done

