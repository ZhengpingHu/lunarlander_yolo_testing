import os
import csv
import numpy as np
import gymnasium as gym
from PIL import Image
import cv2

# === 配置 ===
OUTPUT_DIR = "../testing"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
META_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

# 采样参数
NUM_SAMPLES   = 600
Y_RANGE       = (5.5, 8.0)            # 起始 y 范围（米）
X_RANGE       = (0.0, 10.0)           # 起始 x 范围（米）
ANGLE_RANGE   = (-np.pi/4, np.pi/4)   # 起始角度范围（弧度）

# 火焰圆圈模拟参数
NUM_CIRCLES   = 30                    # 每帧火焰圆圈数
RADIUS        = 2                     # 圆圈固定半径（像素）
COLOR_RANGE   = ((200,255),(100,180),(0,80))  # R,G,B 随机区间
OFFSET_RADIUS = 320                     # 圆心相对喷口点的最大扰动（像素）

# 确保 FixedLander-v3 环境已注册
import fixed_env  # 注册环境

# 世界坐标转像素坐标
def world_to_pixel(x, y, scale=30.0, img_h=400):
    return int(x * scale), int(img_h - y * scale)

os.makedirs(IMAGES_DIR, exist_ok=True)

with open(META_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename","init_x","init_y","init_angle","variant"])

    for idx in range(NUM_SAMPLES):
        # 1. 随机采样状态
        x   = np.random.uniform(*X_RANGE)
        y   = np.random.uniform(*Y_RANGE)
        ang = np.random.uniform(*ANGLE_RANGE)

        # 2. Reset 并获取原图
        env = gym.make(
            "FixedLander-v3",
            render_mode="rgb_array",
            init_x=x,
            init_y=y,
            init_angle=ang
        )
        obs, _ = env.reset()
        img = env.render()  # RGB ndarray
        env.close()

        # 3. 计算地面高度图（白色像素）
        h, w = img.shape[:2]
        ground_mask = np.all(img == [255,255,255], axis=2)
        ground_y_map = np.full(w, h)
        for col in range(w):
            ys = np.where(ground_mask[:, col])[0]
            if ys.size > 0:
                ground_y_map[col] = ys.min()

        # 4. 保存无火焰版本
        name0 = f"samp{idx:04d}_no_flame_x{x:.3f}_y{y:.3f}_ang{ang:.3f}.png"
        path0 = os.path.join(IMAGES_DIR, name0)
        cv2.imwrite(path0, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.writerow([name0, x, y, ang, "no_flame"])

        # 5. Reset 并读取 Lander 位置
        env = gym.make(
            "FixedLander-v3",
            render_mode="rgb_array",
            init_x=x,
            init_y=y,
            init_angle=ang
        )
        obs, _ = env.reset()
        lander = env.unwrapped.lander
        cx, cy = lander.position
        env.close()

        # 6. 转像素并进行 LEG_DOWN 偏移
        px, py = world_to_pixel(cx, cy)
        from gymnasium.envs.box2d.lunar_lander import LEG_DOWN
        down_pix = int(LEG_DOWN)
        nozzle_x, nozzle_y = px, py + down_pix

        # 7. 叠加抗锯齿圆圈模拟火焰，仅在地面以上产生
        flame = img.copy()
        for _ in range(NUM_CIRCLES):
            # 随机偏移圆心
            dx = np.random.randint(-OFFSET_RADIUS, OFFSET_RADIUS+1)
            dy = np.random.randint(-OFFSET_RADIUS, OFFSET_RADIUS+1)
            cx2 = np.clip(nozzle_x + dx, 0, w-1)
            cy2 = np.clip(nozzle_y + dy, 0, h-1)
            # 仅在地面 y 之上绘制
            if cy2 < ground_y_map[int(cx2)]:
                col = (
                    int(np.random.randint(*COLOR_RANGE[0])),
                    int(np.random.randint(*COLOR_RANGE[1])),  
                    int(np.random.randint(*COLOR_RANGE[2]))
                )
                cv2.circle(flame, (cx2, cy2), RADIUS, col, -1, lineType=cv2.LINE_AA)

        # 8. 保存有火焰版本
        name1 = f"samp{idx:04d}_with_flame_x{x:.3f}_y{y:.3f}_ang{ang:.3f}.png"
        path1 = os.path.join(IMAGES_DIR, name1)
        cv2.imwrite(path1, cv2.cvtColor(flame, cv2.COLOR_RGB2BGR))
        writer.writerow([name1, x, y, ang, "with_flame"])

print(f"Testing dataset generated: {2*NUM_SAMPLES} images in {IMAGES_DIR}")
