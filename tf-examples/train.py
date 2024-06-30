import os
from pathlib import Path
from sys import platform


import keras
import tensorflow as tf

from keras import layers
from keras import ops
import matplotlib.pyplot as plt

from utils import windows_to_wsl_path


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.cast(image, tf.float32)


def create_dataset(folder_path: Path) -> tf.data.Dataset:
    """"""

    print(folder_path)
    paths = []
    for file in folder_path.iterdir():
        # print(file.suffix)
        if file.suffix in [".jpg", ".png"]:
            paths.append(str(file))

    image_paths = tf.ragged.constant(paths)
    data = tf.data.Dataset.from_tensor_slices(image_paths)
    data = data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(lambda x: x / 255.0)
    data = data.ragged_batch(32)
    return data


if __name__ == '__main__':

    print(tf.config.list_physical_devices('GPU'))

    celeb_folder = Path(r"C:\tmp\DCGAN\Examples\img_align_celeba")
    # celeb_folder = Path(r"C:\tmp\DCGAN\open3d_shapes\raveled")
    if platform == 'linux':
        celeb_folder = windows_to_wsl_path(windows_path=celeb_folder)

    dataset = create_dataset(folder_path=celeb_folder)
    for x in dataset:
        plt.axis("off")
        plt.imshow((x.numpy() * 255).astype("int32")[0])
        break

    plt.show()
    print(len(dataset))
