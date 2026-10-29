# -*- coding: utf-8 -*-
"""
低算力平台上的任意角度中文识别 Demo

    15°×24 旋转投票 + PP-OCR   |   MaixCAM / MaixPy v4

问题：物块上的"冠 / 亚 / 季"朝向随机，OCR 模型只能认正着的字；
      MaixCAM 算力有限，端到端检测模型又需要大量标注数据。

做法：不是让模型对旋转不变，而是把角度喂给它——
      对 ROI 按 15° 步长转 24 次覆盖 360°，各自送 OCR，投票取结果。
      24 个采样点把姿态误差压到 7.5° 以内，中间角度不再是盲区。

依赖：/root/models/pp_ocr.mud
      /maixapp/share/font/ppocr_keys_v1.ttf（缺了只影响屏幕显示，不影响识别）

作者：滕鑫胜 (vega-thrive)
"""

from maix import image, camera, display, app, time, nn
import math

# ---------------- 参数 ----------------
OCR_MODEL = "/root/models/pp_ocr.mud"
TARGETS = {"冠": 11, "亚": 22, "季": 33}   # 字符 -> 结果码
ALIAS = {"业": "亚"}                        # 模型把"亚"读成"业"，别名补偿
STEP = 15                                   # 旋转步长：360/15 = 24 张
ROI_SCALE = 2.5                             # ROI = 检测框 × 2.5（防旋转裁剪，同时裁掉背景）
MIN_VOTES = 2                               # 判定所需最少票数

W, H = 320, 240
CAM_FPS = 30

ocr = None
font_ok = False


def ensure_ocr():
    """懒加载模型。缺模型时直接抛异常，让调用方决定怎么处理。"""
    global ocr, font_ok
    if ocr is None:
        ocr = nn.PP_OCR(OCR_MODEL)
        try:
            image.load_font("ppocr", "/maixapp/share/font/ppocr_keys_v1.ttf", size=20)
            image.set_default_font("ppocr")
            font_ok = True
        except Exception:
            font_ok = False   # 字体缺失只影响屏显，识别照常
    return ocr


def normalize_points(pts):
    """兼容 [[x,y],...] 与 [x0,y0,x1,y1,...] 两种 to_list 返回"""
    if pts and isinstance(pts[0], (list, tuple)):
        return pts
    return list(zip(pts[0::2], pts[1::2]))


def rotate_points(cx, cy, angle_deg, w, h):
    """三点仿射：把 (0,0)(w,0)(0,h) 围绕中心旋转 angle_deg 度。

    MaixPy 的 affine 需要源点/目的点两组扁平坐标；只给三个点即可确定
    整个旋转+平移变换，比逐像素计算快得多。
    """
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    src_flat, dst_flat = [], []
    for x, y in ((0, 0), (w, 0), (0, h)):
        tx, ty = x - cx, y - cy
        src_flat += [int(x), int(y)]
        dst_flat += [int(tx * cos_a - ty * sin_a + cx),
                     int(tx * sin_a + ty * cos_a + cy)]
    return src_flat, dst_flat


def detect_box(model, img_bgr):
    """先用 0° 原图定位最大的候选框（任意角度都能出框）"""
    best, best_area = None, 0
    for obj in model.detect(img_bgr):
        pts = normalize_points(obj.box.to_list())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        if bw * bh > best_area:
            best_area = bw * bh
            best = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2, bw, bh)
    return best


def vote_ocr(model, roi):
    """对 ROI 做 24 次旋转识别，返回 (字符, 票数, 总票数统计)"""
    counts = {"冠": 0, "亚": 0, "季": 0}
    rw, rh = roi.width(), roi.height()
    ocr_w, ocr_h = model.input_width(), model.input_height()

    for angle in range(0, 360, STEP):
        src, dst = rotate_points(rw / 2, rh / 2, angle, rw, rh)
        rotated = roi.affine(src, dst, width=rw, height=rh)
        # OCR 模型要求 BGR 输入：先缩放，再画进 BGR 画布
        small = rotated.resize(ocr_w, ocr_h)
        bgr = image.Image(ocr_w, ocr_h, image.Format.FMT_BGR888)
        bgr.draw_image(0, 0, small)
        for obj in model.detect(bgr):
            ch = ALIAS.get(obj.char_str(), obj.char_str())
            if ch in counts:
                counts[ch] += 1

    # 票数最高、达到下限、且严格领先（避免 2:2 平票乱判）
    winner, win_votes = None, 0
    for ch, v in counts.items():
        if v > win_votes:
            winner, win_votes = ch, v
    second = max((v for ch, v in counts.items() if ch != winner), default=0)
    if win_votes >= MIN_VOTES and win_votes > second:
        return winner, win_votes, counts
    return None, win_votes, counts


def main():
    cam = camera.Camera(W, H, fps=CAM_FPS)
    disp = display.Display()
    cam.skip_frames(20)

    try:
        model = ensure_ocr()
    except Exception as e:
        print("[ERROR] 模型加载失败:", e)
        print("请确认 %s 存在" % OCR_MODEL)
        return

    print("就绪：把印有 冠/亚/季 的物块放到画面里")
    while not app.need_exit():
        t0 = time.ticks_ms()
        img = cam.read()
        img_bgr = img.to_format(image.Format.FMT_BGR888)

        box = detect_box(model, img_bgr)
        if box is not None:
            cx, cy, bw, bh = box
            size = max(60, int(max(bw, bh) * ROI_SCALE))
            if size <= W and size <= H:
                x = max(0, min(cx - size // 2, W - size))
                y = max(0, min(cy - size // 2, H - size))
                roi = img.crop(x, y, size, size)
                img.draw_rect(x, y, size, size, image.COLOR_YELLOW, thickness=1)

                ch, votes, counts = vote_ocr(model, roi)
                ty = y - 22 if y >= 22 else y + size + 2   # 别画到画面外
                if ch:
                    code = TARGETS[ch]
                    img.draw_rect(x, y, size, size, image.COLOR_GREEN, thickness=2)
                    if font_ok:
                        img.draw_string(x + 2, ty, "%s  %d/%d" % (ch, votes, 24),
                                        image.COLOR_GREEN)
                    else:
                        img.draw_string(x + 2, ty, "code %d  %d/24" % (code, votes),
                                        image.COLOR_GREEN)
                    print("识别: %s (code=%d) 票数 %d/24  %s" % (ch, code, votes, counts))
                else:
                    print("未达判定阈值:", counts)

        fps = 1000.0 / max(1, time.ticks_ms() - t0)
        img.draw_string(4, 4, "%.1f fps" % fps, image.COLOR_WHITE)
        disp.show(img)

    print("exit")


if __name__ == "__main__":
    main()
