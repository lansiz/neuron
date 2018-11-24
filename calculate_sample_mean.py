# !/usr/bin/env python
#  -*- coding: utf-8 -*-

import sys
import numpy as np
# print "脚本名：", sys.argv[0]
accu_l = []
for i in range(1, len(sys.argv)):
    accu_l.append(float(sys.argv[i]))
# print "脚本名：", sys.argv[0]
print np.array(accu_l).mean()

