import numpy as np
import matplotlib.pyplot as plt

original = plt.imread('ball.png') 

albedo = plt.imread('ball_albedo.png')
shading = plt.imread('ball_shading.png')

new_color = np.zeros(albedo.shape)
new_color[:, :, 1] = albedo[:, :, 1] / albedo[:, :, 1].max()

reconstructed = np.zeros(albedo.shape)
for i in range(albedo.shape[-1]):
    reconstructed[:, :, i] = new_color[:, :, i] * shading

fig, ax = plt.subplots(1, 2)
ax[0].imshow(original)
ax[1].imshow(reconstructed)

plt.show()