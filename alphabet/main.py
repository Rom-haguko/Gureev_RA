import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation
from pathlib import Path

def get_hole_count(area):
    dims = area.image.shape
    padded_img = np.zeros((dims[0] + 2, dims[1] + 2))
    padded_img[1:-1, 1:-1] = area.image
    inverted_img = np.logical_not(padded_img)
    labeled_img = label(inverted_img)
    return np.max(labeled_img) - 1

def count_full_columns(area):
    return np.all(area.image, axis=0).sum()

def more_lines_on_left(area):
    column_flags = area.image.mean(axis=0) == 1
    mid = len(column_flags) // 2
    return np.sum(column_flags[:mid]) > np.sum(column_flags[mid:])

def classify_symbol(area):
    if np.all(area.image):
        return "-"
    holes = get_hole_count(area)
    if holes == 2:
        left_heavy = more_lines_on_left(area)
        cy, cx = area.centroid_local
        cx /= area.image.shape[1]
        if left_heavy and cx < 0.44:
            return "B"
        return "8"
    elif holes == 1:
        cy, cx = area.centroid_local
        cx /= area.image.shape[1]
        cy /= area.image.shape[0]
        if more_lines_on_left(area):
            if cx > 0.4 or cy > 0.4:
                return "D"
            return "P"
        if abs(cx - cy) < 0.04:
            return "0"
        return "A"
    else:
        if count_full_columns(area) >= 3:
            return "1"
        if area.eccentricity < 0.5:
            return "*"
        inv_area = ~area.image
        inv_area = binary_dilation(inv_area, np.ones((3, 3)))
        labeled_inv = label(inv_area, connectivity=1)
        region_count = np.max(labeled_inv)
        if region_count == 2:
            return "/"
        if region_count == 4:
            return "X"
        return "W"

img_path = Path(__file__).parent / "symbols.png"
img = plt.imread(img_path)[:, :, :-1]
gray_img = img.mean(axis=2)
binary_img = gray_img > 0
labeled_img = label(binary_img)
areas = regionprops(labeled_img)

symbol_tally = {}

plt.figure()
for i, area in enumerate(areas, 1):
    symbol = classify_symbol(area)
    symbol_tally[symbol] = symbol_tally.get(symbol, 0) + 1

print(symbol_tally)
