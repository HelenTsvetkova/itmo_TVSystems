import cv2
import numpy as np
import statistics
import os

save_dir_path = "./results/"
if not os.path.exists(save_dir_path):
   os.makedirs(save_dir_path)

dir_path = "./data/"

images_paths = [dir_path + '/' + path for path in os.listdir(dir_path)]
images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in images_paths]
assert(len(images) == 10)

images_mean = images[0]
for i in range(1, len(images)):
    alpha = 1.0/(i + 1)
    beta = 1.0 - alpha
    images_mean = cv2.addWeighted(images[i], alpha, images_mean, beta, 0.0)

image = np.asarray(images_mean[:,:])
rows, cols = image.shape

for roi_width in range(1, 4):
    for roi_height in range(1, 4):
        image_bright = image
        sum_array = []
        for r in range(0, rows, roi_height):
            for c in range(0, cols, roi_width):
                roi_sum = image_bright[r:r+roi_height, c:c+roi_width].sum()
                if(roi_sum > 255):
                    roi_sum = 255
                sum_array.append(roi_sum)
                image_bright[r:r+roi_height, c:c+roi_width] = np.full((roi_height, roi_width), roi_sum)

        print("roi - ", roi_height, roi_width)
        print("mean sum = ", statistics.median(sum_array))
        print("min sum - ", min(sum_array))
        print("max sum - ", max(sum_array))
        result_name = "./results/width-" + str(roi_width) + "_height-" + str(roi_height) + ".bmp"
        cv2.imwrite(result_name, image_bright)
