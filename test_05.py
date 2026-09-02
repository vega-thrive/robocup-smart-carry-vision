# 任务05控制变量 Step 0：裸OCR观察（无旋转/无跟踪/无阈值/无串口）
# 目的：确认模型在真实画面上能否读出文本、框的是什么
# 观察协议：正放(近/中/远) → 斜30°/45°/90°，每帧打印 框尺寸/文本/分数
from maix import image, camera, display, app, time, nn

ocr = nn.PP_OCR("/root/models/pp_ocr.mud")
cam = camera.Camera(320, 240, fps=30)
disp = display.Display()
cam.skip_frames(30)

font_ok = False
try:
    image.load_font("ppocr", "/maixapp/share/font/ppocr_keys_v1.ttf", size=20)
    image.set_default_font("ppocr")
    font_ok = True
except Exception:
    print("ppocr 字体缺失，仅影响屏幕中文显示")

printed_dir = False

while not app.need_exit():
    img = cam.read()
    t = time.ticks_ms()

    img_bgr = img.to_format(image.Format.FMT_BGR888)  # OCR模型要求BGR输入
    objs = ocr.detect(img_bgr)

    if not printed_dir and objs:
        printed_dir = True
        print("[成员] OCR_Object:", [a for a in dir(objs[0]) if not a.startswith('_')])

    if not objs:
        print("[无框]")
    for obj in objs:
        raw_pts = obj.box.to_list()
        img.draw_keypoints(raw_pts, image.COLOR_RED, 4, -1, 1)
        # 框尺寸
        if raw_pts and isinstance(raw_pts[0], (list, tuple)):
            xs = [p[0] for p in raw_pts]
            ys = [p[1] for p in raw_pts]
        else:
            xs = raw_pts[0::2]
            ys = raw_pts[1::2]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        text = obj.char_str()
        score = getattr(obj, "score", -1)
        print("[框] 尺寸=%dx%d 文本='%s' 分数=%s" % (bw, bh, text, score))
        if font_ok:
            img.draw_string(obj.box.x4, obj.box.y4, "[%dx%d] '%s' %.2f" % (bw, bh, text, score if score is not None else -1), image.COLOR_RED)

    dt = time.ticks_ms() - t
    if dt <= 0:
        dt = 1
    img.draw_string(5, 5, "fps:%.1f time:%dms" % (1000.0 / dt, dt), image.COLOR_RED, scale=1)
    disp.show(img)
