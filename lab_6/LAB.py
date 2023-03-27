import cv2
import numpy as np
import statistics
import os

save_dir_path = "./results/"
if not os.path.exists(save_dir_path):
   os.makedirs(save_dir_path)

dir_path = "./data/"
image_name = "1.bmp"

image_cv2 = cv2.imread(dir_path + image_name, cv2.IMREAD_GRAYSCALE)
image = np.asarray(image_cv2[:,:])
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
