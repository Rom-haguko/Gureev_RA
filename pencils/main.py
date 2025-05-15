import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.filters import sobel, threshold_otsu
from scipy.ndimage import binary_fill_holes

def find_pencil(region, size):
    h, w = region.image.shape
    minr, minc, maxr, maxc = region.bbox
    y, x = region.centroid
    y_local = y - minr
    x_local = x - minc
    w_safe = w if w > 0 else 1
    h_safe = h if h > 0 else 1
    x_norm = x_local / w_safe
    y_norm = y_local / h_safe

    center_ok = (0.4 < x_norm < 0.6) and (0.4 < y_norm < 0.6)

    diag = (h**2 + w**2) ** 0.5
    size_ok = (diag > size / 2) and (diag < size)

    perim_ratio = region.perimeter / diag if diag > 0 else 0
    shape_ok = 2.48 < perim_ratio < 5.52

    long_boi = (region.perimeter ** 2) / region.area > 33.33 if region.area > 0 else False

    return center_ok and size_ok and shape_ok and long_boi

count_all = 0

min_region_area = 100

for i in range(1, 13):
    print(f'Обработка изображения {i}...')
    try:
        image = plt.imread(f"./images/img ({i}).jpg").mean(axis=2)
    except FileNotFoundError:
        print(f"Ошибка: файл ./images/img ({i}).jpg не найден. Пропускаем.")
        continue

    s = sobel(image)

    thresh = threshold_otsu(s)
    binary_mask = np.zeros_like(s, dtype=np.uint8)
    binary_mask[s >= thresh / 2] = 1

    filled_holes = binary_fill_holes(binary_mask, np.ones((3, 3)))

    labeled = label(filled_holes)

    all_regions = regionprops(labeled)

    filtered_regions = [
        region for region in all_regions
        if region.area is not None and region.area > min_region_area
    ]

    if not filtered_regions:
        print(f'На изображении {i} найдено 0 шт.')
        continue

    count = 0
    size = np.min(image.shape)

    for region in filtered_regions:
        if find_pencil(region, size):
            count += 1

    print(f'На изображении {i} найдено {count} шт.')
    count_all += count

print(f'\nНа всех изображениях найдено {count_all} шт.')
