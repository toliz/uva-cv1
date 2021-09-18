import numpy as np
import matplotlib.pyplot as plt

original = plt.imread('ball.png') 

albedo = plt.imread('ball_albedo.png')
shading = plt.imread('ball_shading.png')

reconstructed = np.zeros(albedo.shape)
for i in range(albedo.shape[-1]):
    reconstructed[:, :, i] = albedo[:, :, i] * shading

fig, ax = plt.subplots(1, 4)
ax[0].imshow(original)
ax[1].imshow(albedo)
ax[2].imshow(shading)
ax[3].imshow(reconstructed)

plt.show()