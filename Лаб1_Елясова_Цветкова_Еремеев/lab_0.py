import os
from pathlib import Path
import cv2
import numpy as np

save_dir_path = str(Path().absolute()) + '/mean/'
if not os.path.exists(save_dir_path):
   os.makedirs(save_dir_path)

save_dir_path = str(Path().absolute()) + '/roi1/'
if not os.path.exists(save_dir_path):
   os.makedirs(save_dir_path)

save_dir_path = str(Path().absolute()) + '/roi2/'
if not os.path.exists(save_dir_path):
   os.makedirs(save_dir_path)

save_dir_path = str(Path().absolute()) + '/roi3/'
if not os.path.exists(save_dir_path):
   os.makedirs(save_dir_path)

# 1. фокусная дистанция
f_distances = [0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.2, 1.5, 2, 4, 8, 'Бесконечность']

# 2. усреднение 10-ти фотографий и поиск среднекрадратичного отклонения
for f_dist in f_distances:
    # load images
    dir_path = str(Path().absolute()) + '/' + str(f_dist)

    images_paths = [dir_path + '/' + path for path in os.listdir(dir_path)]
    images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in images_paths]
    
    images_num = len(images)
    assert(images_num == 10)

    # mean of images
    images_mean = images[0]
    for i in range(1, images_num):
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        images_mean = cv2.addWeighted(images[i], alpha, images_mean, beta, 0.0)

    cv2.imwrite(str(Path().absolute()) + '/mean/' + str(f_dist) + '.png', images_mean)

    # std of image
    image_std = 0.0
    for i in range(1, images_num):
        image_cur = images[i].astype(float) / 255.0
        image_std += ((image_cur[:,:] - images_mean)**2).sum() / (image_cur.shape[0]*image_cur.shape[1])
    
    image_std = np.sqrt(image_std/images_num)
    print('Std (' + str(f_dist) + ') = ' + str(image_std))

# 3. вырезать фрагменты изображения    
for f_dist in f_distances:
    image = cv2.imread(str(Path().absolute()) + '/mean/' + str(f_dist) + '.png')

    # крупные детали
    x, y, w, h = (430, 600, 600, 400)
    image_roi = image[y:y+h,x:x+w]
    cv2.imwrite(str(Path().absolute()) + '/roi1/' + str(f_dist) + '.png', image_roi)

    # мелкие детали
    x, y, h, w = (400, 40, 400, 400)
    image_roi = image[y:y+h,x:x+w]
    cv2.imwrite(str(Path().absolute()) + '/roi2/' + str(f_dist) + '.png', image_roi)

    # крупные и мелкие детали
    x, y, h, w = (40, 40, 400, 400)
    image_roi = image[y:y+h,x:x+w]
    cv2.imwrite(str(Path().absolute()) + '/roi3/' + str(f_dist) + '.png', image_roi)


for roi_dir in ['/roi1', '/roi2', '/roi3']:
    dir_path = str(Path().absolute()) + roi_dir

    for f_dist in f_distances:
        print("! f = " + str(f_dist))

        image_path = dir_path + '/' + str(f_dist) + '.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_array = np.asarray( image[:,:] )

        diff = np.diff(image_array.reshape((1, image_array.size)))
        print(diff.sum() / image_array.size)

        rows, cols = image_array.shape
        row_diff = np.zeros((1, cols - 1))
        for row in image_array:
            row_diff += np.diff(row, 1)
        
        print(row_diff.sum() / image_array.size)
            
        col_diff = np.zeros((1, rows - 1))
        for col in image_array.transpose():
            col_diff += np.diff(col, 1)
        
        print(col_diff.sum() / image_array.size)

        all_diff = 0
        for i in range(0, rows):
            for j in range(0, cols - 1):
                all_diff += abs( image_array[i][j] - image_array[i][j + 1] )

        print(all_diff / image_array.size)
        


'''
В системах автофокусировки аналоговой телевизионной камеры,
в качестве характеристики управления используется 
детальность многоградационного изображения D.

При использовании цифровой телевизионной камеры
детальность многоградационного изображения можно
рассчитать как вдоль строк, так и вдоль столбцов.

значение детальности Dx = sum(|U(i+1, j) - U(i,j)|)
значение детальности Dy = sum(|U(i, j + 1) - U(i,j)|)
'''
