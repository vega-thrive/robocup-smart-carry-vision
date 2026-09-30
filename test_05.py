# test 验证版：15°旋转ROI + det投票识别冠/亚/季（320×240）
# 复刻用户验证过的流程：ROI旋转走MaixPy affine，resize到模型输入，RGB→BGR，det，计数投票
# 流程：预览定位(4s) → 单帧取ROI(框×2.5外扩) → 24角度旋转投票 → 显示结果 → 循环
from maix import camera, display, image, nn, app, time
import math

model = "/root/models/pp_ocr.mud"
ocr = nn.PP_OCR(model)
cam = camera.Camera(320, 240, fps=30)
disp = display.Display()
cam.skip_frames(30)

try:
    image.load_font("ppocr", "/maixapp/share/font/ppocr_keys_v1.ttf", size=20)
    image.set_default_font("ppocr")
    font_ok = True
except Exception:
    font_ok = False

TARGETS = ("冠", "亚", "季")
ALIAS = {"业": "亚"}
STEP = 15

def normalize_points(pts):
    if pts and isinstance(pts[0], (list, tuple)):
        return pts
    return list(zip(pts[0::2], pts[1::2]))

def get_rotate_points_flat(cx, cy, angle_deg, w, h):
    """三点仿射旋转：源(0,0)(w,0)(0,h)绕中心旋转angle_deg"""
    src_pts = [(0, 0), (w, 0), (0, h)]
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    src_flat = []
    dst_flat = []
    for x, y in src_pts:
        tx = x - cx
        ty = y - cy
        rx = tx * cos_a - ty * sin_a + cx
        ry = tx * sin_a + ty * cos_a + cy
        src_flat.append(int(x))
        src_flat.append(int(y))
        dst_flat.append(int(rx))
        dst_flat.append(int(ry))
    return src_flat, dst_flat

def largest_box(img_bgr):
    """0° det找最大框，返回 (cx, cy, bw, bh) 或 None"""
    best = None
    best_area = 0
    for obj in ocr.detect(img_bgr):
        pts = normalize_points(obj.box.to_list())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        area = bw * bh
        if area > best_area:
            best_area = area
            best = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2, bw, bh)
    return best

# ============ 主循环：预览定位 → 投票识别 → 显示结果 ============
while not app.need_exit():
    # --- 预览定位（4秒） ---
    located = None
    t_end = time.ticks_ms() + 4000
    while time.ticks_ms() < t_end and not app.need_exit():
        img = cam.read()
        img_bgr = img.to_format(image.Format.FMT_BGR888)
        located = largest_box(img_bgr)
        if located is not None:
            cx, cy, bw, bh = located
            img.draw_rect(cx - bw // 2 - 5, cy - bh // 2 - 5, bw + 10, bh + 10,
                          image.Color.from_rgb(0, 255, 0), thickness=2)
        img.draw_string(5, 5, "preview locate...", image.Color.from_rgb(255, 0, 0), scale=1)
        disp.show(img)

    if located is None:
        print("未定位到字符框，重新预览")
        continue

    # --- 单帧采集 + ROI裁剪（框中心 ×2.5 外扩，夹紧到画面内） ---
    img = cam.read()
    cx, cy, bw, bh = located
    roi_size = int(max(bw, bh) * 2.5)
    if roi_size < 60:
        roi_size = 60
    x = max(0, cx - roi_size // 2)
    y = max(0, cy - roi_size // 2)
    x = min(x, 320 - roi_size)
    y = min(y, 240 - roi_size)
    if x < 0 or y < 0:
        print("ROI越界，重新预览")
        continue
    roi = img.crop(x, y, roi_size, roi_size)

    # --- 24角度旋转投票 ---
    t0 = time.ticks_ms()
    counts = {c: 0 for c in TARGETS}
    rw = roi.width()
    rh = roi.height()
    ocr_w = ocr.input_width()
    ocr_h = ocr.input_height()
    for a in range(0, 360, STEP):
        src, dst = get_rotate_points_flat(rw / 2, rh / 2, a, rw, rh)
        rot = roi.affine(src, dst, width=rw, height=rh)
        ocr_input = rot.resize(ocr_w, ocr_h)
        ocr_input_bgr = image.Image(ocr_w, ocr_h, image.Format.FMT_BGR888)
        ocr_input_bgr.draw_image(0, 0, ocr_input)
        for obj in ocr.detect(ocr_input_bgr):
            ch = ALIAS.get(obj.char_str(), obj.char_str())
            if ch in counts:
                counts[ch] += 1

    sweep_ms = time.ticks_ms() - t0
    winner = None
    for c in TARGETS:
        if counts[c] >= 1 and (winner is None or counts[c] > counts[winner]):
            winner = c

    print("=== 投票结果 (耗时%dms) ===" % sweep_ms)
    for c in TARGETS:
        print("%s: %d 次" % (c, counts[c]))
    print("判定: %s" % (winner if winner else "无"))

    # --- 显示结果 2.5秒 ---
    res_img = image.Image(disp.width(), disp.height())
    res_img.draw_string(10, 10, "=== 投票结果 ===", image.Color.from_rgb(0, 255, 0), scale=1)
    ypos = 40
    for c in TARGETS:
        res_img.draw_string(10, ypos, "%s: %d次" % (c, counts[c]),
                            image.Color.from_rgb(255, 255, 255), scale=1)
        ypos += 25
    if winner:
        if font_ok:
            res_img.draw_string(10, ypos, "判定: %s" % winner, image.Color.from_rgb(0, 255, 0), scale=2)
        else:
            res_img.draw_string(10, ypos, "WIN", image.Color.from_rgb(0, 255, 0), scale=2)
    else:
        res_img.draw_string(10, ypos, "无", image.Color.from_rgb(255, 0, 0), scale=2)
    disp.show(res_img)
    t_end = time.ticks_ms() + 2500
    while time.ticks_ms() < t_end and not app.need_exit():
        time.sleep_ms(50)
