import socket
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label

host = "84.237.21.36"
port = 5152
beat = None

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))
    plt.ion()
    plt.figure()
    i = 0
    for i in range(10):
        i+= 1
        sock.send(b"get")
        bts = recvall(sock, 40002)
        beat = b"nope"

        im1 = np.frombuffer(bts[2:40002], dtype="uint8").reshape(bts[0], bts[1])

        labeled_im1 = label(im1 > 0)
        regions = regionprops(labeled_im1)
        if len(regions) == 2:
            reg1 = regions[0]
            reg2 = regions[1]

            x_reg1, y_reg1 = reg1.centroid
            x_reg2, y_reg2 = reg2.centroid

            distance = np.sqrt((x_reg2 - x_reg1) ** 2 + (y_reg2 - y_reg1) ** 2)

            sock.send(f"{distance:.1f}".encode())
            print(sock.recv(10))
            sock.send(b"beat")
            beat = sock.recv(10)
            plt.clf()
            plt.subplot(121)
            plt.imshow(im1)
            plt.pause(1)
