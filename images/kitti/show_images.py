import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

Il = imread('./image_0/000070_10.png')
Ir = imread('./image_1/000070_10.png')
Id = imread('./disp_noc/000070_10.png')

print(np.max(Id))

Il = np.array(Il)
Ir = np.array(Ir)
Id = np.array(Id).astype('float') / 256.0
valid = Id > 0.

f, ax = plt.subplots(2, 2, figsize=(12, 5))

ax[0][0].imshow(Il, cmap='gray')
ax[0][0].set_title('Left')

ax[1][0].imshow(Ir, cmap='gray')
ax[1][0].set_title('Right')

ax[0][1].imshow(Id)
ax[0][1].set_title('Disparity')

ax[1][1].imshow(valid)
ax[1][1].set_title('Valid Pixels')

plt.show()
