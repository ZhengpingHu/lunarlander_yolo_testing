import os
import csv
import numpy as np
import gymnasium as gym
import cv2
from PIL import Image

# === Configuration ===
OUTPUT_DIR = "testing_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
META_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

# Sampling settings
NUM_SAMPLES = 400               # number of (x, angle) samples
INIT_Y = 6.0                    # fixed initial y position
X_RANGE = (0.0, 10.0)           # world x range for sampling
ANGLE_RANGE = (-np.pi/4, np.pi/4)  # angle range (radians)

# Flame drawing settings
NUM_CIRCLES = 20                # number of circles per flame
RADIUS_RANGE = (2.5, 3.0)          # pixel radius range for circles
COLOR_RANGE = ((200, 255), (50, 150), (0, 50))  # ranges for R, G, B
OFFSET_RANGE = (-45, 45)        # pixel offset from nozzle position

# Ensure FixedLander-v3 is registered
import fixed_env

# Utility: convert Box2D coordinates to pixel
def world_to_pixel(x, y, scale=30.0, img_h=400):
    return int(x * scale), int(img_h - y * scale)

# Create output directories
os.makedirs(IMAGES_DIR, exist_ok=True)

# Write metadata
with open(META_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "init_x", "init_y", "init_angle", "variant"])

    # Generate samples
    for idx in range(NUM_SAMPLES):
        x = np.random.uniform(*X_RANGE)
        angle = np.random.uniform(*ANGLE_RANGE)

        # Instantiate environment for this sample with custom init parameters
        env = gym.make(
            "FixedLander-v3",
            render_mode="rgb_array",
            init_x=x,
            init_y=INIT_Y,
            init_angle=angle
        )
        obs, info = env.reset()
        base_img = env.render()  # RGB ndarray

        # Save no_flame variant
        name_nf = f"samp{idx:04d}_no_flame_x{round(x,3)}_ang{round(angle,3)}.png"
        path_nf = os.path.join(IMAGES_DIR, name_nf)
        Image.fromarray(base_img).save(path_nf)
        writer.writerow([name_nf, x, INIT_Y, angle, "no_flame"])

        # Prepare flame variant by drawing circles
        flame_img = base_img.copy()
        # Compute nozzle position at approximate lander bottom center
        lander = env.unwrapped.lander
        cx, cy = lander.position
        px, py = world_to_pixel(cx, cy)
        # Draw random circles as flame
        for _ in range(NUM_CIRCLES):
            radius = np.random.randint(*RADIUS_RANGE)
            dx = np.random.randint(*OFFSET_RANGE)
            dy = np.random.randint(*OFFSET_RANGE)
            color = (
                np.random.randint(*COLOR_RANGE[0]),
                np.random.randint(*COLOR_RANGE[1]),
                np.random.randint(*COLOR_RANGE[2])
            )
            center = (px + dx, py + abs(dy) + radius)  # shift downwards
            cv2.circle(flame_img, center, radius, color, -1)

        # Save with_flame variant
        name_f = f"samp{idx:04d}_with_flame_x{round(x,3)}_ang{round(angle,3)}.png"
        path_f = os.path.join(IMAGES_DIR, name_f)
        cv2.imwrite(path_f, cv2.cvtColor(flame_img, cv2.COLOR_RGB2BGR))
        writer.writerow([name_f, x, INIT_Y, angle, "with_flame"])

        # Clean up env for this sample
        env.close()

print(f"Testing dataset generated: {2 * NUM_SAMPLES} images saved to {IMAGES_DIR}")
