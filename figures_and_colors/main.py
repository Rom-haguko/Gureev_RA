import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv

def analyze_shades(colors, std_mult=1.0):
    print()
    diffs = np.diff(sorted(colors))
    splits = np.where(diffs > np.std(diffs) * std_mult)[0]
    num_shades = len(splits) + 1
    prev = 0
    for i, split in enumerate(splits, start=1):
        count = split - prev + 1
        print(f"Оттенок {i}: {count} объектов")
        prev = split + 1
    last_count = len(colors) - prev
    print(f"Оттенок {num_shades}: {last_count} объектов")


image = plt.imread('balls_and_rects.png')

gray = image.mean(axis=2)
binary = gray > 0

labeled = label(binary)
regions = regionprops(labeled)

count_rec = 0
count_cir = 0
color_rec = list()
color_cir  = list()


for region in regions:
    h, w = region.image.shape
    area = h * w
    is_rectangle_area = (region.area == area)

    ratio = region.minor_axis_length / region.major_axis_length
    is_circle_ratio = (ratio > 0.9)

    y, x = region.centroid
    hue = rgb2hsv(image[int(y), int(x)])[0]

    if is_circle_ratio and not is_rectangle_area:
        count_cir += 1
        color_cir.append(hue)

    else:
        count_rec += 1
        color_rec.append(hue)
colors_all = [rgb2hsv(image[int(r.centroid[0]), int(r.centroid[1])])[0] for r in regions]


print(f'Количество кругов: {count_cir}')
print(f'Количество прямоугольников: {count_rec}')
print(f'Количество всех фигур: {len(colors_all)}')

print("\nОБЩИЕ ОТТЕНКИ: ")
analyze_shades(colors_all, std_mult=1.0)
print("\nОТТЕНКИ КРУГОВ: ")
analyze_shades(color_cir, std_mult=2.0)
print("\nОТТЕНКИ ПРЯМОУГОЛЬНИКОВ: ")
analyze_shades(color_rec, std_mult=2.0)


