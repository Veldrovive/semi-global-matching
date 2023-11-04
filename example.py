from pathlib import Path
import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
from stereo_disparity import stereo_disparity
from stereo_disparity_fast import stereo_disparity_fast

current_file_path = Path(__file__).parent
output_dir = current_file_path / 'output'
output_dir.mkdir(exist_ok=True)
example_images_dir = current_file_path / 'images'

cones = example_images_dir / 'cones' / 'cones_image_02.png', example_images_dir / 'cones' / 'cones_image_06.png'
kitti = example_images_dir / 'kitti' / 'image_0' / '000070_10.png', example_images_dir / 'kitti' / 'image_1' / '000070_10.png'
mars_1 = example_images_dir / 'mars' / 'spirit-01l.png', example_images_dir / 'mars' / 'spirit-01r.png'
mars_6 = example_images_dir / 'mars' / 'spirit-06l.png', example_images_dir / 'mars' / 'spirit-06r.png'
teddy = example_images_dir / 'teddy' / 'teddy_image_02.png', example_images_dir / 'teddy' / 'teddy_image_06.png'

example_images = {
    'cones': cones,
    'kitti': kitti,
    'mars_1': mars_1,
    'mars_6': mars_6,
    'teddy': teddy
}

for name, (left, right) in example_images.items():
    print(f'Processing {name}...')
    left_im = imread(left, mode='F')
    right_im = imread(right, mode='F')

    # Define the bounding box to be the entire image
    bounding_box = np.array([[0, left_im.shape[1] - 1], [0, left_im.shape[0] - 1]])

    plot_vertical = left_im.shape[1] / left_im.shape[0] > 1.5

    disparity_fast = stereo_disparity_fast(left_im, right_im, bounding_box, maxd=55)
    disparity_high_quality = stereo_disparity(left_im, right_im, bounding_box, maxd=55)

    fig = plt.figure()
    if plot_vertical:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    else:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    
    ax2.imshow(disparity_high_quality, cmap='gray')
    ax2.set_title('High quality')
    ax1.imshow(disparity_fast, cmap='gray')
    ax1.set_title('Fast')

    fig.suptitle(name)
    plt.savefig(output_dir / f'{name}.png')
    plt.close(fig)

    input_fig = plt.figure()
    if plot_vertical:
        input_ax1 = input_fig.add_subplot(211)
        input_ax2 = input_fig.add_subplot(212)
    else:
        input_ax1 = input_fig.add_subplot(121)
        input_ax2 = input_fig.add_subplot(122)

    input_ax2.imshow(right_im, cmap='gray')
    input_ax2.set_title('Right')
    input_ax1.imshow(left_im, cmap='gray')
    input_ax1.set_title('Left')

    input_fig.suptitle(name)
    plt.savefig(output_dir / f'{name}_input.png')
    plt.close(input_fig)