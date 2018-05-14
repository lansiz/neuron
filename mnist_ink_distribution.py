import numpy as np
import mnist
import matplotlib.pyplot as plt

imgs = mnist.get_imgs_by_number()
inks = np.array([i[1].sum() / 255 for i in imgs])
print(inks.mean(), inks.std(), inks.max(), inks.min())

fig, ax = plt.subplots(1, 1, figsize=(3, 2))
# for t in range(trails):
ax.hist(inks, bins=30)
ax.tick_params(labelsize=8)
ax.set_xlim(0, 784)
ax.annotate('784', (400, 4000), fontsize=8, color='black')

# ax.set_ylim(0, 1)
plt.savefig('mnist_ink_distribution.png')
plt.show()
