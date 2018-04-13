#!/bin/sh

nums="0 1 2 3 4 5 6 7 8 9"
for num in $nums; do
python simu_09.py -m $num -s 500 -c 2 -r 3 -i 6000 &
python simu_09.py -m $num -s 500 -c 2 -r 3 -i 6000 &
python simu_09.py -m $num -s 500 -c 2 -r 3 -i 6000 &
done

