import os
from pathlib import Path
from copy import deepcopy
import time

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


def get_box() -> (list, list):
    """"""
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    return points, lines


def get_pyramid() -> (list, list):
    points = [
        [0, 0, 0],  # Point 0: Base corner
        [1, 0, 0],  # Point 1: Base corner
        [1, 1, 0],  # Point 2: Base corner
        [0, 1, 0],  # Point 3: Base corner
        [0.5, 0.5, 1]  # Point 4: Apex
    ]

    lines = [
        [0, 1],  # Base edge
        [1, 2],  # Base edge
        [2, 3],  # Base edge
        [3, 0],  # Base edge
        [0, 4],  # Side edge
        [1, 4],  # Side edge
        [2, 4],  # Side edge
        [3, 4]  # Side edge
    ]
    return points, lines


def get_circle() -> (list, list):

    radius = 1.0  # Radius of the circle
    num_segments = 36  # Number of segments in the approximation polygon

    # Calculate the angle between each segment
    angle_increment = 2 * np.pi / num_segments

    # Define the points and lines
    points = [[radius * np.cos(i * angle_increment), radius * np.sin(i * angle_increment), 0] for i in
              range(num_segments)]
    lines = [[i, (i + 1) % num_segments] for i in range(num_segments)]
    return points, lines


def get_cylinder() -> (list, list):
    """"""
    radius = 3.0  # Radius of the circles
    height = 2.0  # Height of the cylinder
    num_segments = 18  # Number of segments for the circle approximation

    # Angle increment for each segment in radians
    angle_increment = 2 * np.pi / num_segments

    # Generate points for the bottom and top circles
    bottom_points = [[radius * np.cos(i * angle_increment), radius * np.sin(i * angle_increment), 0] for i in
                     range(num_segments)]
    top_points = [[x, y, height] for x, y, _ in
                  bottom_points]  # Shift the bottom points up by 'height' to get the top points

    # Combine the points into a single list
    points = bottom_points + top_points

    # Generate lines for the bottom and top circles
    bottom_lines = [[i, (i + 1) % num_segments] for i in range(num_segments)]
    top_lines = [[i + num_segments, ((i + 1) % num_segments) + num_segments] for i in range(num_segments)]

    # Generate lines connecting corresponding points on the top and bottom
    side_lines = [[i, i + num_segments] for i in range(num_segments)]

    # Combine all lines into a single list
    lines = bottom_lines + top_lines + side_lines
    return points, lines

def create_cone():
    radius = 3.0  # Radius of the base circle
    height = 4.0  # Height of the cone from the base to the apex
    num_segments = 16  # Number of segments for the circle approximation

    # Angle increment for each segment in radians
    angle_increment = 2 * np.pi / num_segments

    # Generate points for the base circle
    base_points = [[radius * np.cos(i * angle_increment), radius * np.sin(i * angle_increment), 0] for i in
                   range(num_segments)]

    # Define the apex point
    apex_point = [0, 0, height]  # The apex is above the center of the base circle

    # Combine the base points and the apex into a single list
    # The apex is added at the end of the list, so its index will be 'num_segments'
    points = base_points + [apex_point]

    # Generate lines for the base circle
    base_lines = [[i, (i + 1) % num_segments] for i in range(num_segments)]

    # Generate lines connecting the base points to the apex
    side_lines = [[i, num_segments] for i in range(num_segments)]  # 'num_segments' is the index of the apex point

    # Combine the base circle lines and the side lines into a single list
    lines = base_lines + side_lines
    return points, lines


def get_diamond() -> (list, list):
    """"""
    # Parameters
    base_side = 1.0  # Side length of the square base
    height = 0.5  # Height from the base to each apex

    # Points for the square base (shared by both pyramids)
    points = [
        [-base_side / 2, -base_side / 2, 0],  # Lower left corner
        [base_side / 2, -base_side / 2, 0],  # Lower right corner
        [base_side / 2, base_side / 2, 0],  # Upper right corner
        [-base_side / 2, base_side / 2, 0]  # Upper left corner
    ]

    # Adding the lower apex and upper apex points
    points.append([0, 0, -height])  # Lower apex, index 4
    points.append([0, 0, height])  # Upper apex, index 5

    # Lines for the lower pyramid
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Base square
        [0, 4], [1, 4], [2, 4], [3, 4]  # Sides connecting to the lower apex
    ]

    # Lines for the upper pyramid
    lines += [
        [0, 5], [1, 5], [2, 5], [3, 5]  # Sides connecting to the upper apex
    ]
    return points, lines


def create_shape():
    print("Let's draw a box using o3d.geometry.LineSet.")
    # points, lines = get_box()
    # points, lines = get_pyramid()
    # points, lines = get_circle()
    # points, lines = get_cylinder()
    # points, lines = create_cone()
    points, lines = get_diamond()

    colors = [[1, 1, 1] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def main():
    # shape = 'circle'
    output_path = Path(fr"C:\tmp\DCGAN\open3d_shapes\raveled")
    output_path.mkdir(exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=128, height=128)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # RGB values for black

    line_set = create_shape()
    # o3d.visualization.draw_geometries([line_set])
    vis.add_geometry(line_set)

    # Set the initial view point
    # Adjust this value to change the distance to the object

    # Define the rotation axis (Z-axis in this case) and rotation speed
    axes = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 1, 1],
    ]
    for i in range(len(axes)):

        rotation_axis = np.array(axes[i])
        # rotation_axis[i] = 1

        angles = np.linspace(0, 1 * np.pi, 360 * 1)  # 360 steps for smoothness

        # Main loop for the animation
        for _, angle in enumerate(angles):

            copy_lineset = deepcopy(line_set)
            R = copy_lineset.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            copy_lineset.rotate(R, center=copy_lineset.get_center())

            # if idx % 25 == 0:
            vis.clear_geometries()
            vis.add_geometry(copy_lineset)

            # view_control = vis.get_view_control()
            # view_control.set_zoom(1.0)

            vis.poll_events()
            vis.update_renderer()
            # time.sleep(0.01)
            # depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            # plt.imsave(output_path / f"box_{idx}.png", np.asarray(image).resize(128, 128))
            idx = len(os.listdir(output_path))
            plt.imsave(output_path / f"{idx:06d}.png", np.asarray(image)[:, 25:153, :])
            # print()



if __name__ == '__main__':
    main()
