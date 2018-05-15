from __future__ import print_function
import numpy as np
import multiprocessing
from nn import NeuralNetwork
from gene import Gene
import strengthen_functions
import matplotlib.pyplot as plt

z32 = [0.05292, 0.054963333333333336, 0.057166666666666664, 0.061926666666666665, 0.06640666666666667, 0.07101, 0.07130333333333333, 0.08081333333333333, 0.08631333333333334, 0.09218, 0.10107333333333333, 0.10443, 0.12069, 0.14466666666666667, 0.17233, 0.17787333333333333, 0.22957666666666668, 0.2859366666666667, 0.41469666666666666, 0.58163]
z12_0 =  [1.00137, 1.7688033333333333, 8.39283, 27.8103, 22.521893333333335, 19.383286666666667, 16.65192, 12.759093333333333, 12.144473333333334, 11.202873333333333, 9.294406666666667, 9.117966666666666, 9.81901, 6.509196666666667, 5.717076666666666, 6.889453333333333, 7.58722, 5.746833333333333, 5.1651533333333335, 5.243193333333333]
z12_1 =  [1.0032433333333333, 1.77597, 8.543093333333333, 29.282396666666667, 23.920883333333332, 18.078553333333332, 14.634096666666666, 12.312273333333334, 13.405926666666666, 11.41938, 9.649266666666668, 8.952246666666667, 7.949516666666667, 8.374426666666666, 7.1676166666666665, 6.70697, 6.13884, 5.658416666666667, 5.887953333333333, 6.5523]
z12_2 =  [1.0050033333333332, 1.7611733333333333, 7.424113333333334, 27.241316666666666, 22.13812, 18.01734, 15.84008, 14.026093333333334, 12.580776666666667, 10.902733333333334, 10.217556666666667, 8.397053333333334, 8.016276666666666, 8.37199, 5.757263333333333, 7.089296666666667, 6.4315, 5.585353333333333, 5.426853333333334, 6.14954]
z12_3 =  [1.0014133333333333, 1.7223833333333334, 6.99441, 27.436023333333335, 22.923183333333334, 17.39077, 13.53542, 14.421996666666667, 11.344636666666666, 9.742166666666666, 9.968473333333334, 8.615196666666666, 9.22595, 7.140653333333334, 5.54126, 6.099893333333333, 6.971536666666666, 4.525923333333333, 4.85128, 4.48393]
z30 = [38.96604, 21.321703333333332, 15.731356666666667, 12.274, 9.87439, 9.244723333333333, 7.79822, 6.990146666666667, 6.56207, 5.980463333333334, 5.1596166666666665, 4.9173, 4.802043333333334, 4.52573, 4.16089, 3.9634, 3.7491433333333335, 3.693, 3.36698, 3.15627]
z15 = [0.25055333333333335, 0.26543333333333335, 0.2903833333333333, 0.31592, 0.34609, 0.3740233333333333, 0.43795666666666666, 0.4755, 0.5555766666666667, 0.6635533333333333, 0.82956, 1.0846633333333333, 1.3489666666666666, 1.1072866666666668, 1.29723, 1.0187333333333333, 1.3080066666666668, 0.8331966666666667, 0.83666, 0.99235]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(9, 4))
# xs1 = np.linspace(0, 1, 100)
# ys1 = [pf(x) for x in xs1]
xs = np.linspace(0, 1, 20)
ax1.plot(xs, z32)
ax2.plot(xs, z12_0)
ax2.plot(xs, z12_1)
ax2.plot(xs, z12_2)
ax2.plot(xs, z12_3)
ax3.plot(xs, z30)
ax4.plot(xs, z15)
ax1.tick_params(labelsize=8)
ax2.tick_params(labelsize=8)
ax3.tick_params(labelsize=8)
ax4.tick_params(labelsize=8)

# ax.set_ylim(0, 1)
plt.savefig('nn_06_2.png')
plt.show()