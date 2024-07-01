import keras
import cv2
import numpy as np


if __name__ == '__main__':

    latent_dim = 128
    num_images = 5
    generator = keras.models.load_model(r"C:\tmp\DCGAN\models-tf\generator_2.keras")
    for j in range(5):
        # random_latent_vectors = keras.random.normal(
        #     shape=(num_images, latent_dim), seed=keras.random.SeedGenerator(42)
        # )

        latent = np.random.rand(1, 128)
        image = generator(latent).numpy() * 255

        cv2.imshow(f'jaa', image[0].astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



