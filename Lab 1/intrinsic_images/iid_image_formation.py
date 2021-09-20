import numpy as np
import matplotlib.pyplot as plt

# read original image
original = plt.imread('ball.png') 

# read the reflectance (albedo) and shading components
albedo = plt.imread('ball_albedo.png')
shading = plt.imread('ball_shading.png')

# reconstruct the image as I[x, y] = R[x, y] x S[x, y] for every pixel [x, y]
reconstructed = np.zeros(albedo.shape)
for i in range(albedo.shape[-1]):
    reconstructed[:, :, i] = albedo[:, :, i] * shading

# print reconstruction error
print(f'Reconstruction error: { ((reconstructed - original) ** 2).mean() }')

# plot & sava results
fig, ax = plt.subplots(1, 4, num='3. Image Reconstruction')
for axes in ax:
    axes.set_axis_off()

ax[0].imshow(original)
ax[1].imshow(albedo)
ax[2].imshow(shading)
ax[3].imshow(reconstructed)

plt.show()
fig.savefig('results/reconstruction.png', pad_inches = 0.2, bbox_inches='tight')
