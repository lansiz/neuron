#!/bin/sh
echo '' > nn_06.log
python nn_06_propagation_depth_3.py -a 1 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 3 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 5 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 7 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 9 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 11 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 13 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 15 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 17 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 19 >> nn_06.log &
python nn_06_propagation_depth_3.py -a 21 >> nn_06.log &
