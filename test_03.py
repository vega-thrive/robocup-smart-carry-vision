# test_03（学长双指标版）：红/蓝/黄对象识别 + 圆度/矩形度判别
# 输出码：31红圆柱 32红长方体 33蓝圆柱 34蓝长方体 35黄圆锥
# 管线：find_blobs 宽阈值颜色定位 → 黄=圆锥(35)；红/蓝用 blob 原生圆度+矩形度判形状
#      → 中间区不判(等下一帧) → 3帧同码确认 → 打印/画框
# 阈值和形状参数为用户实测验证版
from maix import image, camera, display, app, time
import math

cam = camera.Camera(320, 240, fps=30)
disp = display.Display()
cam.skip_frames(30)

# ============ 颜色阈值（用户实测宽版） ============
TH_RED    = [20, 85, 25, 90, 15, 70]
TH_BLUE   = [20, 85, -40, 10, -50, -10]
TH_YELLOW = [30, 90, 5, 40, 30, 80]
MIN_AREA = 300          # 最小色块像素数

# ============ 形状判别参数（用户验证版） ============
CIRCLE_ROUNDNESS_MIN = 0.85   # 圆度 ≥ 此值 且 矩形度 ≥ 此值 = 圆柱
RECTANGULARITY_MIN = 0.65     # 矩形度下限（旋转矩形圆度虽高但矩形度低 → 不判圆柱）
# 赛场上物体只可能是圆柱/长方体：不是圆 = 长方体（无中间区犹豫）

MAX_CONFIDENCE = 3
CODE_MAP = {("Red", "cyl"): 31, ("Red", "box"): 32,
            ("Blue", "cyl"): 33, ("Blue", "box"): 34}
YELLOW_CODE = 35

# ============ 工具 ============
def classify_shape(blob):
    """圆度+矩形度双指标（blob原生数据，无cv2）。
    圆度≥0.85且矩形度≥0.65 = 圆柱；其余一律长方体"""
    area = blob.pixels()
    per = blob.perimeter()
    w = blob[2]
    h = blob[3]
    roundness = (4 * math.pi * area) / (per * per) if per > 0 else 0.0
    rectangularity = area / (w * h) if w * h > 0 else 0.0
    if roundness >= CIRCLE_ROUNDNESS_MIN and rectangularity >= RECTANGULARITY_MIN:
        return ("cyl", roundness, rectangularity)
    return ("box", roundness, rectangularity)

# ============ 主循环 ============
last_code = None
confidence = 0
prev_conf = 0
while not app.need_exit():
    img = cam.read()
    img.gaussian(2)   # 平滑降噪（学长版本验证）
    t = time.ticks_ms()

    # 1) 颜色定位：三色各取最大色块，再取全局最大
    best = None
    for name, th in (("Red", TH_RED), ("Blue", TH_BLUE), ("Yellow", TH_YELLOW)):
        blobs = img.find_blobs([th], pixels_threshold=MIN_AREA)
        if blobs:
            b = max(blobs, key=lambda x: x.pixels())
            if best is None or b.pixels() > best[1].pixels():
                best = (name, b)

    color = None
    shape = None
    code = 0
    cx = cy = 0
    rnd = rect = 0.0
    if best is not None:
        color, blob = best
        bx, by, bw, bh = blob[0], blob[1], blob[2], blob[3]
        cx, cy = blob.cx(), blob.cy()
        img.draw_rect(bx, by, bw, bh, image.COLOR_GREEN, thickness=2)
        img.draw_string(bx, max(0, by - 12), color, image.COLOR_BLACK)
        if color == "Yellow":
            shape = "cone"
            code = YELLOW_CODE
        else:
            shape, rnd, rect = classify_shape(blob)
            code = CODE_MAP.get((color, shape), 0)
        img.draw_string(bx, by + bh + 5, "%s-%s r=%.2f q=%.2f" % (
            color, shape if shape else "?", rnd, rect), image.COLOR_GREEN)

    # 2) 3帧同码确认（状态转换才打印）
    if code != 0 and code == last_code:
        confidence = min(confidence + 1, MAX_CONFIDENCE)
    else:
        last_code = code
        confidence = 1 if code != 0 else 0

    if confidence == MAX_CONFIDENCE and prev_conf < MAX_CONFIDENCE:
        print(">>> 确认: %s-%s code=%d 中心=(%d,%d)" % (color, shape, code, cx, cy))
    prev_conf = confidence

    dt = time.ticks_ms() - t
    if dt <= 0:
        dt = 1
    img.draw_string(5, 5, "fps:%.1f conf:%d/3" % (1000.0 / dt, confidence), image.COLOR_RED, scale=1)
    disp.show(img)
