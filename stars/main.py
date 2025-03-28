import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_erosion

data = np.load("stars.npy")

mask1 = np.array([[1, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0],
                  [1, 0, 0, 0, 1]])

mask2 = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [1, 1, 1, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]])


def count_stars_with_erosion(image):
    eroded_cross = binary_erosion(image, mask1)
    eroded_plus = binary_erosion(image, mask2)

    labeled_cross = label(eroded_cross)
    labeled_plus = label(eroded_plus)

    cross_count = np.max(labeled_cross)
    plus_count = np.max(labeled_plus)

    total_stars = cross_count + plus_count

    return total_stars


print(f"Количество звездочек: {count_stars_with_erosion(data)}")
