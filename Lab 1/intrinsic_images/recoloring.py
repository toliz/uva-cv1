import numpy as np
import matplotlib.pyplot as plt

# read original image
original = plt.imread('ball.png') 

# read the reflectance (albedo) and shading components
albedo = plt.imread('ball_albedo.png')
shading = plt.imread('ball_shading.png')

# ball colour
print(f'Original ball colour: { (albedo.max(axis=(0, 1)) * 256).astype(int) }')

new_color = np.zeros(albedo.shape)                              # zero R, B channels
new_color[:, :, 1] = albedo[:, :, 1] / albedo[:, :, 1].max()    # set non-zero values of G to 1

# reconstruct recoloured image
reconstructed = np.zeros(albedo.shape)
for i in range(albedo.shape[-1]):
    reconstructed[:, :, i] = new_color[:, :, i] * shading

# plot & save results
fig, ax = plt.subplots(1, 2, num='3. Image Recolouring')
for axes in ax:
    axes.set_axis_off()

ax[0].imshow(original)
ax[1].imshow(reconstructed)

plt.show()
fig.savefig('results/recolouring.png', pad_inches = 0.2, bbox_inches='tight')
