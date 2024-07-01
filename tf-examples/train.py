import os
from pathlib import Path
from sys import platform


import keras
import tensorflow
import tensorflow as tf
import cv2
from keras import layers
from keras import ops
import matplotlib.pyplot as plt

from utils import windows_to_wsl_path
from models import create_gan, GANMonitor


def load_image(image_path, target_size: int = 64):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (target_size, target_size))
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
    # celeb_folder = Path(r"C:\tmp\DCGAN\open3d_shapes\raveled2")
    output_path = Path(r"C:\tmp\DCGAN\output")
    if platform == 'linux':
        celeb_folder = windows_to_wsl_path(windows_path=celeb_folder)
        output_path = windows_to_wsl_path(windows_path=output_path)

    # print(output_path)
    dataset = create_dataset(folder_path=celeb_folder)
    epochs = 100
    latent_dim = 128
    gan = create_gan(latent_dim=latent_dim)

    checkpoint_filepath = r'C:\tmp\DCGAN\models-tf\gan.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    gan.fit(
        dataset,
        epochs=epochs,
        callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)],
    )

    print("Ok bro")


    # for x in dataset:
    #     # plt.axis("off")
    #     # image = (x.numpy() * 255).astype("int32")[0]
    #     image = (x.numpy() * 255).astype("int32")[0]
    #     # plt.imshow(image)
    #     cv2.imwrite(str(output_path / 'test.jpg'), image)
    #     break
    #
    # plt.show()



    # print(len(dataset))
