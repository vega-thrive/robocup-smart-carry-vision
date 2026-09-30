from maix import image, camera, display, app, time, uart, nn
import numpy as np 
import cv2
import math

# 初始化摄像头和显示器
cam = camera.Camera(320, 240, fps=30)
disp = display.Display()
cam.skip_frames(30)  # 跳过开头的30帧，以稳定摄像头输出

# 初始化串口
device = "/dev/ttyS0"  # 指定UART设备为/dev/ttyS0
serial0 = uart.UART(device, 115200)  # 初始化UART对象，波特率设置为115200

# 包头和包尾
HEADER1 = 0xFF
HEADER2 = 0xAA
FOOTER1 = 0xFE
FOOTER2 = 0xBB

# 初始化数据项的值
data1 = 0
data2 = 0
data3 = 0
data4 = 0

# ============== 圆检测参数和函数 ==============
MIN_AREA = 850
MAX_AREA = 90000
MARGIN = 2

#Hom区检测
# detector = nn.YOLO11(model="/root/models/banyun666/model_8301.mud")  # 圆/色块检测用OpenCV，不需要YOLO模型

def is_circle(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return 0.15 < circularity < 1.35  # 接近1表示更接近圆形   return 0.65 < circularity < 1.35

def auto_canny(image, sigma=0.33):
    """自动计算Canny边缘检测的阈值"""
    v = np.median(image)
    if v <= 0:
        return cv2.Canny(image, 70, 140)
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

# 椭圆参数范围
MIN_AXIS = 30  # 最小轴长
MAX_AXIS = 300  # 最大轴长
MIN_ASPECT_RATIO = 0.5  # 最小长短轴比例 0.9


def detect_circle(img, MIN_AXIS):
    """霍夫圆检测（对同心圆/圆环比 fitEllipse 稳定）"""
    img_raw = image.image2cv(img, copy=True)  # 转换为OpenCV格式（BGR）
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)  # 灰度转换
    gray = cv2.medianBlur(gray, 5)  # 中值滤波降噪

    # 霍夫圆检测：minDist=最小圆心间距，param2越小越敏感；dp=2 提速约4倍（精度仍够用）
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=2,
                               minDist=40, param1=100, param2=30,
                               minRadius=MIN_AXIS, maxRadius=200)
    if circles is not None:
        x, y, r = circles[0][0]   # 取检测到的第一个（最明显的）圆
        return int(x), int(y)
    return None

# ============== 旧版椭圆拟合法（备份，主循环已改用霍夫圆） ==============
def detect_circle_fitEllipse(img,MIN_AXIS):
    """使用圆拟合检测圆形（支持部分遮挡/变形）"""
    img_raw = image.image2cv(img, copy=True)  # 转换为OpenCV格式（BGR）
    current_pos = 0
    # ==================== 预处理（保留边缘细节） ====================
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)  # 灰度转换
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 轻度高斯模糊降噪
    
    # 自适应Canny边缘检测（动态阈值）
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))  # 降低低阈值（适应暗部）
    upper = int(min(255, (1.0 + 0.33) * v))  # 提高高阈值（适应亮部）
    edged = cv2.Canny(gray, lower, upper)

    # ==================== 轮廓检测（兼容不完整轮廓） ====================
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_ellipse = None
    min_contour_area = 90  # 最小轮廓面积
    
    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 只处理面积大于一定值的轮廓（排除小噪点）
        if area < min_contour_area:
            continue
        
        # 椭圆拟合（需要至少6个点）
        if len(contour) >= 6:
            # 拟合椭圆参数：中心坐标、轴长（长轴，短轴）、旋转角度
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major_axis, minor_axis), angle = ellipse
            
            # 转换为整数
            center_x_ellipse = int(x)
            center_y_ellipse = int(y)
            major_axis = int(major_axis)
            minor_axis = int(minor_axis)
            
            # 确保坐标在图像范围内
            center_x_ellipse = max(0, min(center_x_ellipse, img.width() - 1))
            center_y_ellipse = max(0, min(center_y_ellipse, img.height() - 1))
            
            # 筛选有效的椭圆
            # 1. 检查轴长是否在范围内
            if MIN_AXIS <= major_axis <= MAX_AXIS and MIN_AXIS <= minor_axis <= MAX_AXIS:
                # 2. 检查长短轴比例（避免过于扁平的椭圆）
                aspect_ratio = minor_axis / major_axis if major_axis > minor_axis else major_axis / minor_axis
                if aspect_ratio >= MIN_ASPECT_RATIO:
                    # 选择最接近圆形的椭圆（aspect_ratio接近1）
                    if detected_ellipse is None or abs(1 - aspect_ratio) < abs(1 - detected_ellipse[6]):
                        detected_ellipse = (center_x_ellipse, center_y_ellipse, 
                                           major_axis, minor_axis, angle, aspect_ratio)
                        
                    return center_x_ellipse,center_y_ellipse
# ============== 结束圆检测部分 ==============


def detect_circle_2(img,MIN_AREA):
    """使用圆拟合检测圆形（支持部分遮挡/变形）"""
    img_raw = image.image2cv(img, copy=True)  # 转换为OpenCV格式（BGR）

    # ==================== 预处理（保留边缘细节） ====================
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)  # 灰度转换
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 轻度高斯模糊降噪
    
    # 自适应Canny边缘检测（动态阈值）
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))  # 降低低阈值（适应暗部）
    upper = int(min(255, (1.0 + 0.33) * v))  # 提高高阈值（适应亮部）
    edged = cv2.Canny(gray, lower, upper)

    # ==================== 轮廓检测（兼容不完整轮廓） ====================
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_pos = None  # 最终检测到的圆心坐标

    for contour in contours:
        # 1. 过滤面积过小/过大的轮廓
        area = cv2.contourArea(contour)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        # 2. 圆拟合：使用最小包围圆（OpenCV内置函数）
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius == 0:  # 避免除零错误
            continue

        # 3. 验证拟合圆的有效性（面积比+边缘距离）
        # 计算拟合圆的面积
        circle_area = np.pi * (radius ** 2)
        # 计算轮廓面积与拟合圆面积的比值（越接近1越接近完美圆）
        area_ratio = area / circle_area if circle_area != 0 else 0

        # 计算圆心到轮廓边缘的最小距离（验证是否被遮挡）
        # 方法：计算轮廓上所有点到圆心的距离，取最小值
        contour_points = contour.squeeze(axis=1).astype(np.float32)  # 展平轮廓点
        distances = np.sqrt((contour_points[:, 0] - x) ** 2 + (contour_points[:, 1] - y) ** 2)
        min_distance = np.min(distances) if len(distances) > 0 else 0

        # 4. 综合条件判断（可调整阈值）
        # 条件1：面积比≥0.8（允许20%的变形/遮挡）
        # 条件2：最小边缘距离≥MARGIN（避免边缘被截断）
        # 条件3：圆心在图像有效区域内
        in_bounds = (
            (x - radius > MARGIN) and 
            (y - radius > MARGIN) and 
            (x + radius < img_raw.shape[1] - MARGIN) and 
            (y + radius < img_raw.shape[0] - MARGIN)
        )

        if area_ratio >= 0.6 and min_distance >= MARGIN and in_bounds:
            current_pos = (int(x), int(y))
            break  # 找到第一个符合条件的圆（可根据需求改为保留最大圆）

    return current_pos
# ============== 结束圆检测部分 ==============

# ============== OCR识别部分（任务05：冠/亚/季，支持任意角度） ==============
OCR_MODEL = "/root/models/pp_ocr.mud"
TARGETS = {"冠": 11, "亚": 22, "季": 33}   # 字符 -> 串口码
ALIAS = {"业": "亚"}       # 模型把"亚"读成"业"，别名补偿
STEP = 15                  # 旋转投票步长（15°×24=360°，用户验证通过）
ROI_SCALE = 2.5            # ROI = 检测框 × 2.5（防旋转裁剪 + 裁掉背景）
MIN_VOTES = 2              # 判定所需最少票数

ocr = None
ocr_font_loaded = False
ocr_font_ok = False

def ensure_ocr():
    """懒加载OCR模型：任务05才初始化，避免模型缺失影响其他任务启动"""
    global ocr, ocr_font_loaded, ocr_font_ok
    if ocr is None:
        ocr = nn.PP_OCR(OCR_MODEL)
    if not ocr_font_loaded:
        try:
            image.load_font("ppocr", "/maixapp/share/font/ppocr_keys_v1.ttf", size=20)
            image.set_default_font("ppocr")
            ocr_font_ok = True
        except Exception:
            ocr_font_ok = False  # 字体缺失只影响屏幕显示，不影响识别和串口
        ocr_font_loaded = True
    return ocr

def normalize_points(pts):
    """兼容点对列表和扁平坐标两种to_list返回"""
    if pts and isinstance(pts[0], (list, tuple)):
        return pts
    return list(zip(pts[0::2], pts[1::2]))

def get_rotate_points_flat(cx, cy, angle_deg, w, h):
    """三点仿射旋转：源(0,0)(w,0)(0,h)绕中心旋转angle_deg（MaixPy affine用，用户验证）"""
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
# ============== 结束OCR部分 ==============

# 初始化神经网络分类器
#classifier = nn.Classifier(model="/root/ABC3/model_146973.mud")

# 帧计数器
frame_count = 0
data = b'\x01'  # 初始数据

num = 0
numA = 0
numB = 0
numC = 0
numD = 0

# 任务三配置（新规则对象：红/蓝圆柱、红/蓝长方体、黄圆锥，正俯视识别）
TH_RED    = [20, 85, 25, 90, 15, 70]      # 红色（用户实测宽版）
TH_BLUE   = [20, 85, -40, 10, -50, -10]   # 蓝色
TH_YELLOW = [30, 90, 5, 40, 30, 80]       # 黄色
MIN_AREA = 300                            # 最小色块像素数
CIRCLE_ROUNDNESS_MIN = 0.85               # 圆度下限（圆柱判据）
RECTANGULARITY_MIN = 0.65                 # 矩形度下限（旋转矩形圆度虚高时防误判）
CODE_MAP3 = {("Red", "cyl"): 31, ("Red", "box"): 32,
             ("Blue", "cyl"): 33, ("Blue", "box"): 34}

def classify_shape(blob):
    """圆度+矩形度判别：圆度≥0.85且矩形度≥0.65=圆柱(cyl)，其余=长方体(box)"""
    area = blob.pixels()
    per = blob.perimeter()
    w = blob[2]
    h = blob[3]
    roundness = (4 * math.pi * area) / (per * per) if per > 0 else 0.0
    rectangularity = area / (w * h) if w * h > 0 else 0.0
    if roundness >= CIRCLE_ROUNDNESS_MIN and rectangularity >= RECTANGULARITY_MIN:
        return "cyl"
    return "box"
def find_circles_ABC(data1, data2, data3, data4):
    """打包并发送数据"""
    # 将数据项转换为字节数据，确保值在0-255范围内
    
    def clamp_byte(value):
        """确保值在0-255范围内"""
        return max(0, min(255, value))
    def clamp_uint16(value):
        """确保值在0-65535范围内"""
        return max(0, min(65535, value))
    """打包并发送数据"""
    # 确保数据在有效范围内
    data1 = clamp_byte(data1)
    data2 = clamp_byte(data2)
    data3 = clamp_uint16(data3)
    data4 = clamp_uint16(data4)

    # 构建数据包（uint16 拆分：d3/d4 各 2 字节高+低，坐标不截断）
    packet = bytes([HEADER1, HEADER2]) + \
             data1.to_bytes(1, 'big') + data2.to_bytes(1, 'big') + \
             data3.to_bytes(2, 'big') + data4.to_bytes(2, 'big') + \
             bytes([FOOTER1, FOOTER2])
    # 发送数据
    serial0.write(packet)

    # 打印发送的数据（注释掉以提升帧率）
    # print(f"Sent: {[f'0x{b:02X}' for b in packet]}")
    # print(f"Values: data1={data1}, data2={data2}, data3={data3}, data4={data4}")

# 主循环
while not app.need_exit():
    img = cam.read()
    # 串口接收数据（读1字节，没数据就继续等，不再强制进入任务一）
    data = serial0.read(1)
    time.sleep_ms(5)
    if data is None:
        time.sleep_ms(10)
        continue
    # 任务一：圆检测
    if data == b'\x01':
        while True:
              
            img = cam.read()
            t = time.ticks_ms()  # 获取当前时间戳（毫秒）
            # 在图像中心绘制检测区域
            center_x = img.width() // 2
            center_y = img.height() // 2
            img.draw_rect(center_x , center_y , 6, 6, image.COLOR_RED, thickness=1)
            
            # 使用改进的圆检测算法（椭圆拟合法，帧率比霍夫圆高）
            current_pos = detect_circle_fitEllipse(img,30)
           
            # 初始化数据
            data1 = data2 = data3 = data4 = 0
            
            if current_pos is not None:
                circle_x, circle_y = current_pos
                
                # 检查圆心是否在中心区域
                if (center_x - 3 <= circle_x <= center_x + 3 and 
                    center_y - 3 <= circle_y <= center_y + 3):
                    data2 = 1  # 圆心在中心区域
                    img.draw_cross(circle_x, circle_y, image.COLOR_RED, size=20, thickness=1)
                    img.draw_string(circle_x + 10, circle_y + 10, "1", image.COLOR_RED)
                else:
                    data2 = 0  # 圆心不在中心区域
                    img.draw_cross(circle_x, circle_y, image.COLOR_GREEN, size=20, thickness=1)
                    img.draw_string(circle_x + 10, circle_y + 10, "0", image.COLOR_BLACK)
                
                data1 = 1  # 检测到圆
                data3 = circle_x
                data4 = circle_y
            else:
                data1 = 0  # 未检测到圆
                data2 = 0
                data3 = 320  # 超出 X 边界，让 C 代码进入盲搜而非误判为左上角
                data4 = 240  # 超出 Y 边界，同上
            
            # 发送数据（只有检测到圆 data1==1 才发，避免未检测时发 320→255=0xFF 干扰帧同步）
            if data1 == 1:
                find_circles_ABC(data1, data2, data3, data4)
            
            # 显示图像
            disp.show(img)
            
            # 检查是否退出任务一
            data_in = serial0.read()
            if data_in == b'\x04':
                data = data_in
                break
            
            # 计算并显示帧率
            fps = 1000 / (time.ticks_ms() - t)
            # print(f"FPS: {fps:.1f}, Data: {data1},{data2},{data3},{data4}")
            
            # 增加帧计数器
            frame_count += 1
    # 任务er：圆检测
    #识别冠亚
    elif data == b'\x02':
        while True:
            
            img = cam.read()
            t = time.ticks_ms()  # 获取当前时间戳（毫秒）
            # 在图像中心绘制检测区域
            center_x = img.width() // 2 + 4
            center_y = img.height() // 2 - 60
            img.draw_rect(center_x , center_y , 4, 6, image.COLOR_RED, thickness=1)
            
            # 使用改进的圆检测算法
            current_pos = detect_circle_2(img,50)
           
            # 初始化数据
            data1 = data2 = data3 = data4 = 0
            
            if current_pos is not None:
                circle_x, circle_y = current_pos
                
                # 检查圆心是否在中心区域
                if (center_x - 2 <= circle_x <= center_x + 2 and 
                    center_y - 3 <= circle_y <= center_y + 3):
                    data2 = 1  # 圆心在中心区域
                    img.draw_cross(circle_x, circle_y, image.COLOR_RED, size=20, thickness=1)
                    img.draw_string(circle_x + 10, circle_y + 10, "1", image.COLOR_RED)
                else:
                    data2 = 0  # 圆心不在中心区域
                    img.draw_cross(circle_x, circle_y, image.COLOR_GREEN, size=20, thickness=1)
                    img.draw_string(circle_x + 10, circle_y + 10, "0", image.COLOR_BLACK)
                
                data1 = 1  # 检测到圆
                data3 = circle_x
                data4 = circle_y
            else:
                data1 = 0  # 未检测到圆
                data2 = 0
                data3 = 320  # 超出 X 边界，让 C 代码进入盲搜而非误判为左上角
                data4 = 240  # 超出 Y 边界，同上
            
            # 发送数据（只有检测到圆 data1==1 才发，避免未检测时发 320→255=0xFF 干扰帧同步）
            if data1 == 1:
                find_circles_ABC(data1, data2, data3, data4)
            
            # 显示图像
            disp.show(img)
            
            # 检查是否退出任务一
            data_in = serial0.read()
            if data_in == b'\x04':
                data = data_in
                break
            
            # 计算并显示帧率
            fps = 1000 / (time.ticks_ms() - t)
            # print(f"FPS: {fps:.1f}, Data: {data1},{data2},{data3},{data4}")
            
            # 增加帧计数器
            frame_count += 1
    
    # 任务六：圆检测
    elif data == b'\x06':
        while True:
            
            img = cam.read()
            t = time.ticks_ms()  # 获取当前时间戳（毫秒）
            # 在图像中心绘制检测区域
            center_x = img.width() // 2 - 6
            center_y = img.height() // 2 - 20
            img.draw_rect(center_x , center_y , 6, 6, image.COLOR_RED, thickness=1)
            
            # 使用改进的圆检测算法
            current_pos = detect_circle(img,600)
           
            # 初始化数据
            data1 = data2 = data3 = data4 = 0
            
            if current_pos is not None:
                circle_x, circle_y = current_pos
                
                # 检查圆心是否在中心区域
                if (center_x - 3 <= circle_x <= center_x + 3 and 
                    center_y - 3 <= circle_y <= center_y + 3):
                    data2 = 1  # 圆心在中心区域
                    img.draw_cross(circle_x, circle_y, image.COLOR_RED, size=20, thickness=1)
                    img.draw_string(circle_x + 10, circle_y + 10, "1", image.COLOR_RED)
                else:
                    data2 = 0  # 圆心不在中心区域
                    img.draw_cross(circle_x, circle_y, image.COLOR_GREEN, size=20, thickness=1)
                    img.draw_string(circle_x + 10, circle_y + 10, "0", image.COLOR_BLACK)
                
                data1 = 1  # 检测到圆
                data3 = circle_x
                data4 = circle_y
            else:
                data1 = 0  # 未检测到圆
                data2 = 0
                data3 = 320  # 超出 X 边界，让 C 代码进入盲搜而非误判为左上角
                data4 = 240  # 超出 Y 边界，同上
            
            # 发送数据（只有检测到圆 data1==1 才发，避免未检测时发 320→255=0xFF 干扰帧同步）
            if data1 == 1:
                find_circles_ABC(data1, data2, data3, data4)
            
            # 显示图像
            disp.show(img)
            
            # 检查是否退出任务一
            data_in = serial0.read()
            if data_in == b'\x04':
                data = data_in
                break
            
            # 计算并显示帧率
            fps = 1000 / (time.ticks_ms() - t)
            # print(f"FPS: {fps:.1f}, Data: {data1},{data2},{data3},{data4}")
            
            # 增加帧计数器
            frame_count += 1
    # 任务三：红/蓝/黄对象识别（31红圆柱 32红长方体 33蓝圆柱 34蓝长方体 35黄圆锥）
    elif data == b'\x03':
        last_code = None
        confidence = 0
        while True:
            img = cam.read()
            img.gaussian(2)  # 平滑降噪

            # 颜色定位：三色各取最大色块，再取全局最大
            best = None
            for name, th in (("Red", TH_RED), ("Blue", TH_BLUE), ("Yellow", TH_YELLOW)):
                blobs = img.find_blobs([th], pixels_threshold=MIN_AREA)
                if blobs:
                    b = max(blobs, key=lambda x: x.pixels())
                    if best is None or b.pixels() > best[1].pixels():
                        best = (name, b)

            data1 = data2 = data3 = data4 = 0
            if best is not None:
                color, blob = best
                cx, cy = blob.cx(), blob.cy()
                img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN, thickness=2)
                if color == "Yellow":
                    data1 = 35  # 黄圆锥（颜色唯一）
                else:
                    shape = classify_shape(blob)
                    data1 = CODE_MAP3.get((color, shape), 0)
                data3 = cx
                data4 = cy

            # 3帧同码确认防抖（确认后才发串口）
            if data1 != 0 and data1 == last_code:
                confidence = min(confidence + 1, 3)
            else:
                last_code = data1
                confidence = 1 if data1 != 0 else 0

            if confidence >= 3:
                find_circles_ABC(data1, data2, data3, data4)

            disp.show(img)

            # 检查是否退出任务三
            data_in = serial0.read()
            if data_in == b'\x04':
                data = data_in
                break

            frame_count += 1
    # 任务五：OCR识别冠/亚/季（旋转投票法：ROI 15°×24 旋转 + det计数，停稳后识别）
    elif data == b'\x05':
        try:
            ocr_model = ensure_ocr()
        except Exception as e:
            print("OCR init failed:", e)
            data = None
            continue
        while True:
            img = cam.read()
            img_bgr = img.to_format(image.Format.FMT_BGR888)  # OCR模型要求BGR输入

            # 0° det 定位最大框（任意角度都出框）
            box = None
            best_area = 0
            for obj in ocr_model.detect(img_bgr):
                raw_pts = obj.box.to_list()  # 原始扁平坐标，draw_keypoints要求List[int]
                img.draw_keypoints(raw_pts, image.COLOR_RED, 4, -1, 1)
                pts = normalize_points(raw_pts)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bw = max(xs) - min(xs)
                bh = max(ys) - min(ys)
                area = bw * bh
                if area > best_area:
                    best_area = area
                    box = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2, bw, bh)

            data1 = data2 = data3 = data4 = 0
            if box is not None:
                cx, cy, bw, bh = box
                roi_size = max(60, int(max(bw, bh) * ROI_SCALE))
                if roi_size <= 320 and roi_size <= 240:
                    x = max(0, min(cx - roi_size // 2, 320 - roi_size))
                    y = max(0, min(cy - roi_size // 2, 240 - roi_size))
                    roi = img.crop(x, y, roi_size, roi_size)
                    img.draw_rect(x, y, roi_size, roi_size, image.COLOR_YELLOW, thickness=1)
                    # 15°×24 旋转投票
                    counts = {"冠": 0, "亚": 0, "季": 0}
                    rw = roi.width()
                    rh = roi.height()
                    ocr_w = ocr_model.input_width()
                    ocr_h = ocr_model.input_height()
                    for a in range(0, 360, STEP):
                        src, dst = get_rotate_points_flat(rw / 2, rh / 2, a, rw, rh)
                        rot = roi.affine(src, dst, width=rw, height=rh)
                        ocr_input = rot.resize(ocr_w, ocr_h)
                        ocr_input_bgr = image.Image(ocr_w, ocr_h, image.Format.FMT_BGR888)
                        ocr_input_bgr.draw_image(0, 0, ocr_input)
                        for obj in ocr_model.detect(ocr_input_bgr):
                            ch = ALIAS.get(obj.char_str(), obj.char_str())  # 业→亚
                            if ch in counts:
                                counts[ch] += 1
                        # 扫描中响应0x04退出，不必等满一轮
                        if serial0.read() == b'\x04':
                            data = b'\x04'
                            break
                    if data == b'\x04':
                        break
                    # 判定：票数最高且≥MIN_VOTES且严格领先
                    winner = None
                    win_votes = 0
                    for c, v in counts.items():
                        if v > win_votes:
                            winner = c
                            win_votes = v
                    second = 0
                    for c, v in counts.items():
                        if c != winner and v > second:
                            second = v
                    if winner is not None and win_votes >= MIN_VOTES and win_votes > second:
                        data1 = TARGETS[winner]
                        data3 = cx
                        data4 = cy
                        if ocr_font_ok:
                            img.draw_string(cx + 10, cy + 10, "%s:%d(%d票)" % (winner, data1, win_votes), image.COLOR_RED)
                        else:
                            img.draw_string(cx + 10, cy + 10, "code:%d(%d)" % (data1, win_votes), image.COLOR_RED)
                        find_circles_ABC(data1, data2, data3, data4)
                        print("识别确认:", winner, "票数:", win_votes, "/24")

            disp.show(img)

            # 检查是否退出任务五
            data_in = serial0.read()
            if data_in == b'\x04':
                data = data_in
                break

            frame_count += 1
    # 显示图像
    disp.show(img)
    # 短暂延迟，避免CPU过载
    time.sleep_ms(10)
