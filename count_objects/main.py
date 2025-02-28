import numpy as np

external = np.array([
    [[0, 0],[0, 1]],
    [[0, 0],[1, 0]],
    [[0, 1],[0, 0]],
    [[1, 0],[0, 0]]
])
internal = np.logical_not(external)
cross = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])

def match(a, masks):
    for mask in masks:
        binary_a = (a != 0) # Преобразует в bool
        if np.all(binary_a == mask):
            return True
    return False

def count_objects(image):
    E = 0
    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):  
            sub = image[y : y + 2, x : x + 2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E / 4




img_ex_1 = np.load("example1.npy")
img_ex_2 = np.load("example2.npy")

list_img_ex_2 = [count_objects(img_ex_2 [:,:,0]),count_objects(img_ex_2 [:,:,1]),count_objects(img_ex_2 [:,:,2])]

print(f'The file "example1.npy" has {count_objects(img_ex_1)} objects')
print(f'The file "example2.npy" has {sum(list_img_ex_2)} objects')
