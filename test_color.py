# 色块识别测试脚本（无串口，直接开始识别，测帧率/颜色）
from maix import image, camera, display, app, time

# === 初始化摄像头和显示 ===
cam = camera.Camera(320, 240, fps=30)
disp = display.Display()
cam.skip_frames(30)

# === 色块的颜色配置（按阈值编辑器实测中心值+余量，前两版注释保留） ===
thresholds_1 = [[16, 38, 35, 65, 15, 40]]      # red   中心(26,48,28)
thresholds_2 = [[35, 55, -25, 0, -45, -15]]    # blue  中心(46,-11,-25)  B上限-15与白拉开
thresholds_3 = [[24, 45, -45, -22, 5, 35]]     # green 中心(34,-34,18)
thresholds_4 = [[50, 70, -8, 8, -8, 15]]       # white 中心(55,0,6)     B下限-8
thresholds_5 = [[0, 15, -8, 8, -10, 8]]        # black 中心(6,0,-2)
# 上一版（手测紧阈值，范围太小导致其它颜色难识别）
# thresholds_1 = [[24, 29, 46, 51, 25, 31]]      # red
# thresholds_2 = [[44, 48, -15, -8, -28, -22]]   # blue
# thresholds_3 = [[32, 37, -36, -31, 16, 21]]    # green
# thresholds_4 = [[53, 58, -2, 3, 2, 8]]         # white
# thresholds_5 = [[4, 8, -2, 3, -4, 1]]          # black
# 旧阈值（与 main.py 一致，需要时恢复）
# thresholds_1 = [[0, 80, 40, 80, 10, 80]]      # red
# thresholds_2 = [[20, 70, -20, 5, -40, 0]]     # blue
# thresholds_3 = [[20, 75, -55, -15, 5, 45]]    # green
# thresholds_4 = [[55, 85, -10, 10, -10, 20]]   # white
# thresholds_5 = [[0, 25, -10, 10, -15, 5]]     # black

# 颜色名和颜色码（与 main.py 任务三一致）
COLORS = [
    ("Red",   21, thresholds_1),
    ("Blue",  22, thresholds_2),
    ("Green", 23, thresholds_3),
    ("White", 24, thresholds_4),
    ("Black", 25, thresholds_5),
]

# 中心区域 ROI
ROI = (110, 90, 130, 110)

# === 主循环 ===
while not app.need_exit():
    img = cam.read()
    t = time.ticks_ms()

    # 绘制检测区域
    img.draw_rect(ROI[0], ROI[1], ROI[2], ROI[3], image.COLOR_WHITE)

    largest = None  # (blob, color_name, color_code)

    # 同时检测所有颜色
    for name, code, th in COLORS:
        blobs = img.find_blobs(th, area_threshold=1000, pixels_threshold=1000, roi=ROI)
        if blobs:
            blob = max(blobs, key=lambda b: b.area())
            if largest is None or blob.area() > largest[0].area():
                largest = (blob, name, code)

    img.draw_string(5, 5, "size:%dx%d" % (img.width(), img.height()), image.COLOR_RED, scale=2)

    if largest is not None:
        blob, color_name, color_code = largest
        # 绘制检测框和颜色名
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
        img.draw_string(blob[0], blob[1] - 10, color_name, image.COLOR_BLACK)
        img.draw_string(5, 30, "color:%s(%d)" % (color_name, color_code), image.COLOR_RED, scale=2)
        img.draw_string(5, 55, "area:%d" % blob.area(), image.COLOR_RED, scale=2)
    else:
        img.draw_string(5, 30, "no color", image.COLOR_RED, scale=2)

    dt = time.ticks_ms() - t
    if dt <= 0:
        dt = 1
    fps = 1000.0 / dt
    img.draw_string(5, 80, "fps:%.1f" % fps, image.COLOR_RED, scale=2)

    disp.show(img)
