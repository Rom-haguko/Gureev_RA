import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label


def calculate_vertical_lines(region):
    return np.all(region.image, axis=0).sum()


def extract_features(region):
    area = region.area / region.image.size
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    eccentricity = region.eccentricity
    euler_number = 1 - region.euler_number
    solidity = region.solidity

    aspect_ratio = region.image.shape[1] / region.image.shape[0]

    center_y = int(region.centroid_local[0])
    center_x = int(region.centroid_local[1])
    row_transitions = np.sum(region.image[center_y, :-1] != region.image[center_y, 1:])
    col_transitions = np.sum(region.image[:-1, center_x] != region.image[1:, center_x])

    half_height, half_width = region.image.shape[0] // 2, region.image.shape[1] // 2
    horizontal_symmetry = np.sum(region.image[:half_height, :] == np.flipud(
        region.image[-half_height:, :])) / region.image.size
    vertical_symmetry = np.sum(region.image[:, :half_width] == np.fliplr(
        region.image[:, -half_width:])) / region.image.size

    return np.array([
        area, cy, cx, perimeter, eccentricity, euler_number, solidity,
        aspect_ratio,
        row_transitions, col_transitions, vertical_symmetry, horizontal_symmetry
    ])


def calculate_distance(v1, v2):
    return ((v1 - v2) ** 2).sum() ** 0.5


def identify_symbol(features, trained_templates):
    best_match = "_"
    min_distance = float('inf')
    for symbol, template in trained_templates.items():
        distance = calculate_distance(features, template)
        if distance < min_distance:
            best_match = symbol
            min_distance = distance
    return best_match


main_alphabet = plt.imread("alphabet.png")[:, :, :-1]
main_gray = main_alphabet.mean(axis=2)
main_binary = main_gray > 0
main_labeled = label(main_binary)
main_regions = regionprops(main_labeled)

template_symbols = plt.imread("alphabet-small.png")[:, :, :-1]
template_gray = template_symbols.mean(axis=2)
template_binary = template_gray < 1
template_labeled = label(template_binary)
template_regions = regionprops(template_labeled)

#Создание шаблонов
trained_templates = {
    "A": extract_features(template_regions[2]),
    "B": extract_features(template_regions[3]),
    "8": extract_features(template_regions[0]),
    "0": extract_features(template_regions[1]),
    "1": extract_features(template_regions[4]),
    "W": extract_features(template_regions[5]),
    "X": extract_features(template_regions[6]),
    "*": extract_features(template_regions[7]),
    "-": extract_features(template_regions[9]),
    "/": extract_features(template_regions[8]),
}


symbol_counts = {}
for region in main_regions:
    features = extract_features(region)
    symbol = identify_symbol(features, trained_templates)
    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

print(symbol_counts)
