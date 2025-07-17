#!/usr/bin/env python3
import os
import shutil
import random
import yaml
import numpy as np
import gymnasium as gym
import cv2
from Box2D import b2Vec2
from Box2D.b2 import polygonShape, circleShape
import fixed_env  # 注册 FixedLander-v3 环境

# ==== Config ==== 
POS_SAMPLES    = 6000
NEG_RATIO      = 3
NEG_S_TOTAL    = POS_SAMPLES * 2 * NEG_RATIO
NEG_MTN_RATIO  = 2/3
SPLITS         = {'train': 0.7, 'val': 0.2, 'test': 0.1}
X_RANGE        = (0.0, 10.0)
Y_RANGE        = (5.5,  8.0)
ANGLE_RANGE    = (-np.pi/4, np.pi/4)
NUM_CIRCLES    = 30
RADIUS         = 2
# 火焰颜色基础及波动
FLAME_BASE     = (255, 140, 0)
FLAME_VAR      = (30, 20, 10)
ALPHA_LEVELS   = [1.0, 0.75, 0.5, 0.25]
PAD            = 10
OFFSET_RADIUS  = 320
OUT        = os.path.join(os.getcwd(), 'testing')
IMG_DIR    = os.path.join(OUT, 'images')
LBL_DIR    = os.path.join(OUT, 'labels')
VIZ_DIR    = os.path.join(OUT, 'viz')
YAML_F     = os.path.join(OUT, 'dataset.yaml')
SCALE      = 30.0
IMG_W = IMG_H = None

# 世界坐标 -> 像素
def world_to_pixel(x, y):
    px = int(x * SCALE)
    py = int(IMG_H - y * SCALE)
    px = max(0, min(px, IMG_W-1))
    py = max(0, min(py, IMG_H-1))
    return px, py

def add_fire(img, center_px, center_py, ground_y, bbox=None):
    """
    在 img 上以 (center_px, center_py) 为喷口中心，绘制 NUM_CIRCLES 片火焰。
    ground_y 用来判断只在地面以上绘制；bbox（x1,y1,x2,y2）用来避开关键体区。
    """
    h, w = img.shape[:2]
    # 如果提供了 bbox，则包成 (x1,y1,x2,y2)
    x1 = y1 = x2 = y2 = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox

    for _ in range(NUM_CIRCLES):
        dx = random.randint(-OFFSET_RADIUS, OFFSET_RADIUS)
        dy = random.randint(-OFFSET_RADIUS, OFFSET_RADIUS)
        xx = np.clip(center_px + dx, 0, w-1)
        yy = np.clip(center_py + dy, 0, h-1)

        # 仅在地面以上，且避开关键体框
        if yy < ground_y[int(xx)] and (bbox is None or not (x1 <= xx < x2 and y1 <= yy < y2)):
            a = random.choice(ALPHA_LEVELS)
            if a <= 0:
                continue
            col = (
                int(np.clip(FLAME_BASE[0] + random.randint(-FLAME_VAR[0], FLAME_VAR[0]), 0, 255) * a),
                int(np.clip(FLAME_BASE[1] + random.randint(-FLAME_VAR[1], FLAME_VAR[1]), 0, 255) * a),
                int(np.clip(FLAME_BASE[2] + random.randint(-FLAME_VAR[2], FLAME_VAR[2]), 0, 255) * a),
            )
            cv2.circle(img, (xx, yy), RADIUS, col, -1, cv2.LINE_AA)

    return img




# 提取 5 个关键点（2 顶角 + 2 腿端 + 1 质心）
def extract_keypoints(env):
    lander = env.unwrapped.lander
    center = lander.position
    angle  = lander.angle
    # 船体顶角
    w_h, h_h = 0.5, 0.5
    offsets = [b2Vec2(-w_h, h_h), b2Vec2(w_h, h_h)]
    ca, sa = np.cos(angle), np.sin(angle)
    def rot(v): return b2Vec2(v.x*ca - v.y*sa, v.x*sa + v.y*ca)
    tops = [center + rot(o) for o in offsets]
    # 腿尖端
    tips = []
    for leg in env.unwrapped.legs:
        leg_c = leg.position
        dx = np.sin(angle) * 0.305
        dy = -np.cos(angle) * 0.3
        tips.append(b2Vec2(leg_c.x + dx, leg_c.y + dy))
    return tops + tips + [center]

# 准备目录
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for d in (IMG_DIR, LBL_DIR, VIZ_DIR):
    os.makedirs(d, exist_ok=True)

idx = POS_COUNT = NEG_COUNT = 0

# 预计算地面高度
gt = gym.make('FixedLander-v3', render_mode='rgb_array')
obs, _ = gt.reset()
gf = gt.render()
gt.close()
IMG_H, IMG_W = gf.shape[:2]
mask = np.all(gf == [255,255,255], axis=2)
ground = np.full(IMG_W, IMG_H)
for c in range(IMG_W):
    ys = np.where(mask[:,c])[0]
    ground[c] = ys.min() if ys.size else IMG_H

# 正样本生成
for _ in range(POS_SAMPLES):
    x0,y0 = random.uniform(*X_RANGE), random.uniform(*Y_RANGE)
    a0    = random.uniform(*ANGLE_RANGE)
    # 渲染基础帧
    env = gym.make('FixedLander-v3', render_mode='rgb_array',
                   init_x=x0, init_y=y0, init_angle=a0)
    obs,_ = env.reset()
    frame = env.render()
    env.close()
    # 提取关键点
    env2 = gym.make('FixedLander-v3', render_mode='rgb_array',
                    init_x=x0, init_y=y0, init_angle=a0)
    obs2,_ = env2.reset()
    kp_w = extract_keypoints(env2)
    env2.close()
    if kp_w is None:
        continue
    kp_px = [world_to_pixel(p.x, p.y) for p in kp_w]
    # 丢弃飞出地面
    bad = False
    for x_raw,y_raw in [(x/IMG_W,y/IMG_H) for x,y in kp_px]:
        gx = int(np.clip(x_raw*IMG_W,0,IMG_W-1))
        if y_raw*IMG_H > ground[gx]:
            bad = True
            break
    if bad:
        continue
    # 计算 bbox
    xs, ys_ = zip(*kp_px)
    x1,y1 = max(0,min(xs)-PAD), max(0,min(ys_)-PAD)
    x2,y2 = min(IMG_W,max(xs)+PAD), min(IMG_H,max(ys_)+PAD)
    cx,cy = ((x1+x2)/2)/IMG_W, ((y1+y2)/2)/IMG_H
    bw,bh = (x2-x1)/IMG_W, (y2-y1)/IMG_H
    # 归一化关键点
    norm = [(x/IMG_W, y/IMG_H) for x,y in kp_px]
    processed = []
    for nx,ny in norm:
        v = 2 if 0<=nx<=1 and 0<=ny<=1 else 0
        nx,ny = np.clip(nx,0,1), np.clip(ny,0,1)
        processed += [f"{nx:.6f}", f"{ny:.6f}", str(v)]
    label = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(processed)
    # 纠正旗帜颜色为黄色 (204,204,0)
    flag_mask = ((frame[:,:,2] > 200) & (frame[:,:,1] < 100) & (frame[:,:,0] < 100))
    frame[flag_mask] = (204, 204, 0)
    # 保存无火焰/有火焰
    for flame in (False, True):
        out_img = frame.copy()
        if flame:
            px0,py0 = world_to_pixel(x0,y0)
            for _c in range(NUM_CIRCLES):
                dx,dy = random.randint(-OFFSET_RADIUS,OFFSET_RADIUS), random.randint(-OFFSET_RADIUS,OFFSET_RADIUS)
                xx,yy = np.clip(px0+dx,0,IMG_W-1), np.clip(py0+dy,0,IMG_H-1)
                if yy < ground[int(xx)] and not (x1<=xx<x2 and y1<=yy<y2):
                    a = random.choice(ALPHA_LEVELS)
                    if a > 0:
                        # 橙色基色 + 波动
                        col = (
                            int(np.clip(FLAME_BASE[0] + random.randint(-FLAME_VAR[0], FLAME_VAR[0]), 0, 255) * a),
                            int(np.clip(FLAME_BASE[1] + random.randint(-FLAME_VAR[1], FLAME_VAR[1]), 0, 255) * a),
                            int(np.clip(FLAME_BASE[2] + random.randint(-FLAME_VAR[2], FLAME_VAR[2]), 0, 255) * a)
                        )
                        cv2.circle(out_img, (xx,yy), RADIUS, col, -1, cv2.LINE_AA)
        fn = f"{idx:06d}.png"
        cv2.imwrite(os.path.join(IMG_DIR,fn), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        with open(os.path.join(LBL_DIR,fn.replace('.png','.txt')),'w') as f:
            f.write(label)
        idx += 1
        POS_COUNT += 1

for _ in range(NEG_S_TOTAL):
    # 1) 环境渲染拿地形+旗子
    env = gym.make("FixedLander-v3", render_mode="rgb_array",
                   init_x=random.uniform(*X_RANGE),
                   init_y=random.uniform(*Y_RANGE),
                   init_angle=random.uniform(*ANGLE_RANGE))
    obs,_ = env.reset()
    frame = env.render()
    env.close()

    # 2) 擦掉着陆器（紫色部分）
    mask = (frame[:,:,0] > 80) & (frame[:,:,1] < 50) & (frame[:,:,2] < 50)
    frame[mask] = (0,0,0)

    # 3) 画火焰
    px0 = random.randint(0, IMG_W-1)
    py0 = random.randint(0, IMG_H-1)
    neg = add_fire(frame, px0, py0, ground, bbox=None)

    # 4) 保存
    fn = f"{idx:06d}.png"
    cv2.imwrite(os.path.join(IMG_DIR,fn),
                cv2.cvtColor(neg, cv2.COLOR_RGB2BGR))
    open(os.path.join(LBL_DIR,fn.replace('.png','.txt')), 'w').close()
    idx += 1
    NEG_COUNT += 1



# 写入 dataset.yaml
with open(YAML_F, 'w') as yf:
    yaml.dump({
        'path': OUT,
        'train':'images/train', 'val':'images/val', 'test':'images/test',
        'nc':1, 'names':['lander'], 'keypoints':5,
        'skeleton':[[0,4],[1,4],[2,4],[3,4]]
    }, yf, sort_keys=False)

# 划分 & 可视化
all_imgs = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
random.shuffle(all_imgs)
n = len(all_imgs)
n1 = int(n * SPLITS['train'])
n2 = int(n * SPLITS['val'])
spl = {
    'train': all_imgs[:n1],
    'val':   all_imgs[n1:n1+n2],
    'test':  all_imgs[n1+n2:]
}
for s, files in spl.items():
    imgd = os.path.join(OUT, 'images', s)
    lbld = os.path.join(OUT, 'labels', s)
    os.makedirs(imgd, exist_ok=True)
    os.makedirs(lbld, exist_ok=True)
    for fn in files:
        shutil.move(os.path.join(IMG_DIR,fn), os.path.join(imgd,fn))
        shutil.move(os.path.join(LBL_DIR,fn.replace('.png','.txt')),
                    os.path.join(lbld,fn.replace('.png','.txt')))

for s in spl:
    od = os.path.join(VIZ_DIR, s)
    os.makedirs(od, exist_ok=True)
    for fn in os.listdir(os.path.join(OUT, 'images', s)):
        if not fn.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(OUT, 'images', s, fn))
        h,w = img.shape[:2]
        lbl = os.path.join(OUT, 'labels', s, fn.replace('.png','.txt'))
        if not os.path.exists(lbl):
            continue
        parts = open(lbl).read().strip().split()
        if len(parts) < 5:
            continue
        cx,cy,bw,bh = map(float, parts[1:5])
        kps = list(map(float, parts[5:]))
        x1 = int((cx - bw/2)*w); y1 = int((cy - bh/2)*h)
        x2 = int((cx + bw/2)*w); y2 = int((cy + bh/2)*h)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        for i in range(0, len(kps), 3):
            px,py,v = int(kps[i]*w), int(kps[i+1]*h), int(kps[i+2])
            if v > 0:
                cv2.circle(img, (px,py), 3, (0,0,255), -1)
        cv2.imwrite(os.path.join(od, fn), img)

print(f"Done: {POS_COUNT} positive, {NEG_COUNT} negative, total {POS_COUNT+NEG_COUNT}")
