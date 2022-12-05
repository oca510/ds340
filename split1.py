import skimage.io
import matplotlib.pyplot as plt
import skimage.filters
import random
from PIL import Image
import cv2
import numpy as np
import scipy
import os
#%matplotlib widget

def gaussian_blur(sharp_image, sigma):
    # Filter channels individually to avoid gray scale images
    blurred_image_r = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 0], sigma=sigma)
    blurred_image_g = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 1], sigma=sigma)
    blurred_image_b = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 2], sigma=sigma)
    blurred_image = np.dstack((blurred_image_r, blurred_image_g, blurred_image_b))
    return blurred_image


def uniform_blur(sharp_image, uniform_filter_size):
    # The multidimensional filter is required to avoid gray scale images
    multidim_filter_size = (uniform_filter_size, uniform_filter_size, 1)
    blurred_image = scipy.ndimage.filters.uniform_filter(sharp_image, size=multidim_filter_size)
    return blurred_image

def blur_image_locally(sharp_image, mask, use_gaussian_blur, gaussian_sigma, uniform_filter_size):

    one_values_f32 = np.full(sharp_image.shape, fill_value=1.0, dtype=np.float32)
    sharp_image_f32 = sharp_image.astype(dtype=np.float32)
    sharp_mask_f32 = mask.astype(dtype=np.float32)

    if use_gaussian_blur:
        blurred_image_f32 = gaussian_blur(sharp_image_f32, sigma=gaussian_sigma)
        blurred_mask_f32 = gaussian_blur(sharp_mask_f32, sigma=gaussian_sigma)

    else:
        blurred_image_f32 = uniform_blur(sharp_image_f32, uniform_filter_size)
        blurred_mask_f32 = uniform_blur(sharp_mask_f32, uniform_filter_size)
        
        
    blurred_mask_inverted_f32 = one_values_f32 - blurred_mask_f32
    weighted_sharp_image = np.multiply(sharp_image_f32, blurred_mask_f32)
    weighted_blurred_image = np.multiply(blurred_image_f32, blurred_mask_inverted_f32)
    locally_blurred_image_f32 = weighted_sharp_image + weighted_blurred_image

    locally_blurred_image = locally_blurred_image_f32.astype(dtype=np.uint8)

    return locally_blurred_image

if __name__ == "__main__":
    input = "dataset_final_corre"
    # 输出目录
    output = os.path.join("dataset_final_length_copy_blur_100")

    def save():
        for image_name in os.listdir(input):
            if image_name.endswith('.jpg'):
                print(image_name)
                sharp_image = skimage.io.imread(os.path.join(input, image_name))
                height, width, channels = sharp_image.shape
                sharp_mask = np.full((height, width, channels), fill_value=1)
                h1 = random.randint(100,height)
                w1 = random.randint(100,width)
                sharp_mask[int(h1 / 2): int(h1), int(w1 / 2): int(w1), :] = 0
                result = blur_image_locally(
                    sharp_image,
                    sharp_mask,
                    use_gaussian_blur=True,
                    gaussian_sigma=100,
                    uniform_filter_size=1000)
                im = Image.fromarray(result)
                im.save(os.path.join(output, image_name))
            else:
                pass
    #         plt.imshow(result)
    #         plt.show(
    save()
    print("finished")










