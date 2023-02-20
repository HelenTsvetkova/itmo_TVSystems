import time 
import numpy as np

def compute_diffs(image):

    image_array = np.asarray( image[:,:] )

    # 1 разложение матрицы в вектор
    start_time = time.time()
    diff = np.array([abs(i) for i in np.diff(image_array.reshape((1, image_array.size)))])
    diff = (diff.sum() / image_array.size)
    t1 = start_time - time.time()

    # 2 проход вдоль .. строк?
    start_time = time.time()
    rows, cols = image_array.shape
    row_diff = np.zeros(cols - 1)
    for row in image_array:
        row_diff += [abs(i) for i in np.diff(row, 1)]

    row_diff = (row_diff.sum() / image_array.size)
    t2 = start_time - time.time()
        
    # 3 проход вдоль .. столбцов???
    start_time = time.time()
    col_diff = np.zeros(rows - 1)
    for col in image_array.transpose():
        col_diff += [abs(i) for i in np.diff(col, 1)]

    col_diff = (col_diff.sum() / image_array.size)
    t3 = start_time - time.time()

    # 4 проход по обоим направлениям
    start_time = time.time()
    all_diff = row_diff + col_diff 
    t4 = start_time - time.time() + t2 + t3

    return ([diff, row_diff, col_diff, all_diff], [t1, t2, t3, t4])