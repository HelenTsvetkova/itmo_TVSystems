import os
import time
import statistics
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_diffs

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

save_dir_path = str(Path().absolute()) + '/plots/'
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
    for i in range(0, images_num):
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

# 4. Нахождение производных
diffs, row_diffs, col_diffs, all_diffs = [], [], [], []
t1, t2, t3, t4 = [], [], [], []
for roi_dir in ['/roi1', '/roi2', '/roi3']:
    dir_path = str(Path().absolute()) + roi_dir

    roi_diffs, roi_row_diffs, roi_col_diffs, roi_all_diffs = [], [], [], []
    for f_dist in f_distances:
        image_path = dir_path + '/' + str(f_dist) + '.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        ([diff, row_diff, col_diff, all_diff], [t1_, t2_, t3_, t4_ ]) = compute_diffs(image=image)

        roi_diffs.append(diff)
        roi_row_diffs.append(row_diff)
        roi_col_diffs.append(col_diff)
        roi_all_diffs.append(all_diff)
        t1.append(t1_)
        t2.append(t2_)
        t3.append(t3_)
        t4.append(t4_)

    diffs.append(roi_diffs)
    row_diffs.append(roi_row_diffs)
    col_diffs.append(roi_col_diffs)
    all_diffs.append(roi_all_diffs)

print("1 mean time = " + str(statistics.median(t1)))
print("2 mean time = " + str(statistics.median(t2)))
print("3 mean time = " + str(statistics.median(t3)))
print("4 mean time = " + str(statistics.median(t4)))

#print(row_diffs)
x_ = range(len(f_distances))
f_distances_labels = [str(i) for i in f_distances]
f_distances_labels[11] = '∞'
for i in range(3):
    #fig, ax = plt.subplots()
    #fig.canvas.draw()
    plt.plot(x_, diffs[i], 'r', x_, row_diffs[i], 'g', x_, col_diffs[i], 'b', x_, all_diffs[i], 'purple')
    plt.xticks(x_, f_distances_labels)
    plt.ylabel('Детальность, отн.ед.')
    plt.xlabel('Фокусное расстояние, м')
    plt.title(f'Roi {i+1}')
    plt.grid(True)
    plt.legend(['Vector diff', 'Row diff', 'Col diff', 'All diff'])
    plt.savefig(str(Path().absolute()) + '/plots/roi' + str(i + 1) + '.png')
    plt.close()

'''
ЕЩЁ не сделано:

3.7.
Постройте графики, показывающие время обработки каждого из
способов нахождения детальности от количества элементов: воспользуйтесь
функциями подсчета времени tic (записывает текущее время) и toc (использует
записанное значение для расчета прошедшего времени).
3.8.
Постройте переходную характеристику для различных дальностей
фокусировки.
Выберите фрагмент изображений с четким переходом между темной
деталью и светлым фоном для каждой из изучаемых дистанций фокусировки.
Постройте графики переходных характеристик для каждой из изучаемых
дистанций фокусировки.
'''
