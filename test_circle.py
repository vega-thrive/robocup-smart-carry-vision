# 同心圆识别测试脚本（椭圆拟合法，无串口，测帧率/尺寸/坐标）
from maix import image, camera, display, app, time
import numpy as np
import cv2

# === 初始化摄像头和显示 ===
cam = camera.Camera(320, 240, fps=30)
disp = display.Display()
cam.skip_frames(30)

# === 圆检测参数 ===
MIN_AXIS = 30
MAX_AXIS = 300
MIN_ASPECT_RATIO = 0.5

# === 圆检测：椭圆拟合（fitEllipse） ===
def detect_circle(img, MIN_AXIS):
    img_raw = image.image2cv(img, copy=True)          # MaixPy图 -> OpenCV图
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 自适应 Canny 边缘检测
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(gray, lower, upper)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 90

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        if len(contour) >= 6:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major_axis, minor_axis), angle = ellipse
            center_x_ellipse = int(x)
            center_y_ellipse = int(y)
            major_axis = int(major_axis)
            minor_axis = int(minor_axis)

            center_x_ellipse = max(0, min(center_x_ellipse, img.width() - 1))
            center_y_ellipse = max(0, min(center_y_ellipse, img.height() - 1))

            if MIN_AXIS <= major_axis <= MAX_AXIS and MIN_AXIS <= minor_axis <= MAX_AXIS:
                aspect_ratio = minor_axis / major_axis if major_axis > minor_axis else major_axis / minor_axis
                if aspect_ratio >= MIN_ASPECT_RATIO:
                    return center_x_ellipse, center_y_ellipse
    return None

# === 主循环 ===
while not app.need_exit():
    img = cam.read()
    t = time.ticks_ms()

    center_x = img.width() // 2
    center_y = img.height() // 2
    img.draw_rect(center_x - 3, center_y - 3, 6, 6, image.COLOR_RED, thickness=1)

    current_pos = detect_circle(img, MIN_AXIS)

    img.draw_string(5, 5, "size:%dx%d" % (img.width(), img.height()), image.COLOR_RED, scale=2)

    if current_pos is not None:
        circle_x, circle_y = current_pos
        if (center_x - 3 <= circle_x <= center_x + 3 and
            center_y - 3 <= circle_y <= center_y + 3):
            img.draw_cross(circle_x, circle_y, image.COLOR_RED, size=20, thickness=1)
        else:
            img.draw_cross(circle_x, circle_y, image.COLOR_GREEN, size=20, thickness=1)
        img.draw_string(5, 30, "c:%d,%d" % (circle_x, circle_y), image.COLOR_RED, scale=2)
    else:
        img.draw_string(5, 30, "no circle", image.COLOR_RED, scale=2)

    dt = time.ticks_ms() - t
    if dt <= 0:
        dt = 1
    fps = 1000.0 / dt
    img.draw_string(5, 55, "fps:%.1f" % fps, image.COLOR_RED, scale=2)

    disp.show(img)
