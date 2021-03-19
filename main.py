import os
from cv2 import cv2
import numpy as np
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

from tensorflow.keras.models import load_model


def save_images(samples, name):
    """
    Save generated plt plots as images to a folder.
    """
    num = samples.shape[0]
    for j in range(num):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.imshow(samples[j, ...].reshape(28, 28), cmap='gray')
        plt.xticks([]), plt.yticks([])
        filename = name + j
        if filename < 10:
            filename = "00" + str(filename)
        elif filename < 100:
            filename = "0" + str(filename)
        ax.figure.savefig(f"./img/{filename}.png")
        fig.clf()


def create_images(dcgan_generator, length):
    """
    Create images based on randomly generated noises.
    Length parameter controls the amount of pictures to be generated and
    represents the total length of the generated clip in seconds (if the frame rate is 32 fps).
    """
    subnoises, noises = [], []

    for i in range(length):
        # generating random noises
        subnoises.append(np.random.normal(0, 1, 100).astype(np.float32))

    for i in range(length):
        if i != (length - 1):
            noise = np.linspace(subnoises[i], subnoises[i+1], 32)
            noises.append(noise)
        else:
            noise = np.linspace(subnoises[i], subnoises[0], 32)
            noises.append(noise)

    j = 0
    for i in tqdm(range(length), desc="Generating images"):
        # predicting 32 images based on random noises and saving them
        numbers = dcgan_generator.predict(noises[i])
        save_images(numbers, j)
        j += 32


def create_video(fps=32):
    """
    Make video clip out of previously generated images.
    Change the fps parameter to adjust the frame rate of the video.
    Note that this will affect the total length of the clip.
    """
    img_array = []
    path = os.path.join(os.getcwd(), "img/*.png")
    for filename in glob.glob(path):
        img = cv2.imread(filename)
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('numbers.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    generator = load_model("dcgan_generator.h5")
    create_images(generator, 10)
    create_video()
