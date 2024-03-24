import random

import numpy as np
import cv2


# Parameters
img_height = 400
img_width = 800

# Step 2: Prepare an image for plotting
num_points = 100  # Decreased sampling to 100 points
wave_amplitude = 100
img = np.ones((img_height, img_width, 3), dtype=np.uint8)  # White background
high_fq = 50

for idx in range(100):
    if idx % 2 == 0:
        img = np.ones((img_height, img_width, 3), dtype=np.uint8)  # White background
        wave_frequency = random.randint(1, 10)
    else:
        # high_fq = random.choice([, 99])
        high_fq = 99
        if random.choice([False, True]):
            high_fq = int(high_fq / 2) + wave_frequency
        else:
            high_fq -= wave_frequency

        wave_frequency = high_fq

        # high_fq = 99
        # high_fq -= wave_frequency
        # wave_frequency = high_fq

        # wave_frequency = high_fq
        # high_fq += 1
        # if high_fq > 100:
        #     high_fq = 50

    # Step 1: Generate sinusoidal wave data with reduced sampling
    x = np.linspace(0, 2 * np.pi * wave_frequency, num_points)  # Generate 100 points
    y = np.sin(x) * wave_amplitude
    print(wave_frequency)
    # Draw a line segment between two points

    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )
    for i in reversed(range(1, len(x))):

        if i < (len(x) / 2):
            continue

        # Scale points to fit in the image width
        x_scaled = np.linspace(0, img_width, num_points)

        # Translate points to fit in the image height
        start_point = (int(x_scaled[i - 1]), int(-y[i - 1] + img_height / 2))
        end_point = (int(x_scaled[i]), int(-y[i] + img_height / 2))
        cv2.line(img, start_point, end_point, color, 1)  # Blue color wave

        j = len(x) - i
        x_scaled = np.linspace(0, img_width, num_points)
        # Translate points to fit in the image height
        start_point = (int(x_scaled[j - 1]), int(-y[j - 1] + img_height / 2))
        end_point = (int(x_scaled[j]), int(-y[j] + img_height / 2))
        cv2.line(img, start_point, end_point, color, 1)  # Blue color wave

        # Step 4: Display the image
        cv2.imshow('Sinusoidal Wave - Reduced Sampling', img)
        cv2.waitKey(5)


    # Step 3: Draw the wave with reduced sampling
    # if idx % 2 == 1:
    #     for i in reversed(range(1, len(x))):
    #         # Scale points to fit in the image width
    #         x_scaled = np.linspace(0, img_width, num_points)
    #
    #         # Translate points to fit in the image height
    #         start_point = (int(x_scaled[i - 1]), int(-y[i - 1] + img_height / 2))
    #         end_point = (int(x_scaled[i]), int(-y[i] + img_height / 2))
    #
    #         # Draw a line segment between two points
    #         # color = (
    #         #     random.randint(0, 255),
    #         #     random.randint(0, 255),
    #         #     random.randint(0, 255),
    #         # )
    #         cv2.line(img, start_point, end_point, color, 1)  # Blue color wave
    #
    #         # Step 4: Display the image
    #         cv2.imshow('Sinusoidal Wave - Reduced Sampling', img)
    #         cv2.waitKey(5)
    #
    # else:
    #     for i in range(1, len(x)):
    #         # Scale points to fit in the image width
    #         x_scaled = np.linspace(0, img_width, num_points)
    #
    #         # Translate points to fit in the image height
    #         start_point = (int(x_scaled[i - 1]), int(-y[i - 1] + img_height / 2))
    #         end_point = (int(x_scaled[i]), int(-y[i] + img_height / 2))
    #
    #         # Draw a line segment between two points
    #         # color = (
    #         #     random.randint(0, 255),
    #         #     random.randint(0, 255),
    #         #     random.randint(0, 255),
    #         # )
    #         # Draw a line segment between two points
    #         cv2.line(img, start_point, end_point, color, 1)  # Blue color wave
    #
    #         # Step 4: Display the image
    #         cv2.imshow('Sinusoidal Wave - Reduced Sampling', img)
    #         cv2.waitKey(5)

    # cv2.waitKey(-1)

cv2.destroyAllWindows()
