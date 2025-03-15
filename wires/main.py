import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import (binary_closing, binary_opening,
                                binary_dilation, binary_erosion)

def func_wires(labeled):
    mask = np.array([True, False])
    print(f"Кол-во проводов: {np.max(labeled)}")
    for i in range(1, len(np.unique(labeled))):
        count = 0
        res = binary_erosion(labeled == i,
                             np.ones(3).reshape(3, 1))
        if (res == False).all():
            print(f'{i} провода не существует')
            continue
        for y in range(0, res.shape[0]):
            for x in range(0, res.shape[1] - 1):
                if (res[y, x:x + 2] == mask).all():
                    count += 1
        if count == 0:
            print(f"{i} провод не порван")
            continue
        count += 1
        print(f'{i} Провод поделен на {count} части')



for i in range(1, 7):
    print(f"Результат wires{i}npy.txt:")
    labeled = label(np.load(f"wires{i}npy.txt"))
    func_wires(labeled)
    print()


plt.subplot(231)
plt.title("wires1npy.txt")
plt.imshow(label(np.load("wires1npy.txt")))
plt.subplot(232)
plt.title("wires2npy.txt")
plt.imshow(label(np.load("wires2npy.txt")))
plt.subplot(233)
plt.title("wires3npy.txt")
plt.imshow(label(np.load("wires3npy.txt")))
plt.subplot(234)
plt.title("wires4npy.txt")
plt.imshow(label(np.load("wires4npy.txt")))
plt.subplot(235)
plt.title("wires5npy.txt")
plt.imshow(label(np.load("wires5npy.txt")))
plt.subplot(236)
plt.title("wires6npy.txt")
plt.imshow(label(np.load("wires6npy.txt")))
plt.show()
