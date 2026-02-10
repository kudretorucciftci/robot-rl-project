import numpy as np
import matplotlib.pyplot as plt

def create_robot_image():
    """
    Creates a simple 32x32 pixel art robot image with an alpha channel
    and saves it as 'robot.png'.
    """
    # Create a 32x32 canvas with 4 channels (RGBA)
    canvas = np.zeros((32, 32, 4))

    # Define colors (R, G, B, A)
    grey = [0.5, 0.5, 0.5, 1]
    dark_grey = [0.3, 0.3, 0.3, 1]
    cyan = [0, 1, 1, 1]

    # Head (Grey square)
    canvas[4:12, 12:20] = grey

    # Eyes (Cyan)
    canvas[6:8, 14:16] = cyan
    canvas[6:8, 18:20] = cyan

    # Body (Dark Grey)
    canvas[12:24, 10:22] = dark_grey

    # Arms (Grey)
    canvas[14:18, 4:10] = grey
    canvas[14:18, 22:28] = grey

    # Legs (Grey)
    canvas[24:30, 12:16] = grey
    canvas[24:30, 18:22] = grey

    # Save the image
    plt.imsave("robot.png", canvas)
    print("robot.png created successfully.")

if __name__ == "__main__":
    create_robot_image()
