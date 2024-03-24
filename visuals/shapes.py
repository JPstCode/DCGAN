""""""
import random
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def generate_random_numbers(total, n):
    # Step 2: Generate n random numbers
    random_numbers = [random.random() for _ in range(n)]

    # Step 3: Normalize these numbers to sum to 1
    sum_of_numbers = sum(random_numbers)
    normalized_numbers = [x / sum_of_numbers for x in random_numbers]

    # Step 4: Scale the numbers to sum up to the total
    scaled_numbers = [int(x * total) for x in normalized_numbers]
    diff = total - np.sum(scaled_numbers)
    scaled_numbers[-1] += diff

    return scaled_numbers


def create_col_pattern(canvas: np.ndarray, n_rows: int, n_cols: int, color: bool, add_padding: bool = True) -> np.ndarray:
    """"""
    row_sizes = generate_random_numbers(canvas.shape[0], n_rows)
    col_sizes = generate_random_numbers(canvas.shape[1], n_cols)

    r_start = 0
    for r in range(n_rows):
        c_start = 0
        r_max = r_start + row_sizes[r]
        for c in range(n_cols):
            if color:
                random_color = (
                    np.uint8(random.randint(0, 255)),
                    np.uint8(random.randint(0, 255)),
                    np.uint8(random.randint(0, 255))
                )
            else:
                random_color = np.uint8(random.randint(0, 255))
            c_max = c_start + col_sizes[c]
            padding_min = random.randint(0, int(canvas.shape[1] / 4))
            padding_max = random.randint(int(canvas.shape[1] / 2), canvas.shape[1] - 100)

            canvas[r_start:r_max, c_start:c_max] = random_color
            if add_padding:

                canvas[r_start:r_max, :padding_min] = 0
                canvas[r_start:r_max, padding_max:] = 0

            c_start = c_max
        r_start = r_max

        cv2.imshow('canvas', canvas)
        cv2.waitKey(5)
        time.sleep(0.01)

    return canvas


def main():
    """"""
    canvas_shape_w = 512
    canvas_shape_h = 512
    n_rows = 32
    n_cols = 16

    color = False
    if color:
        canvas = np.zeros((canvas_shape_h, canvas_shape_w, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((canvas_shape_h, canvas_shape_w, 1), dtype=np.uint8)

    for i in range(0, 256):
        canvas = create_col_pattern(canvas=canvas, n_cols=n_cols, n_rows=n_rows, color=color, add_padding=False)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()