from maix import image, camera, display, app, time, uart, nn
import numpy as np 
import cv2
import math
from maix import gpio, pinmap, time
pinmap.set_pin_function("B3", "GPIOB3")
led = gpio.GPIO("GPIOB3", gpio.Mode.OUT)
led.value(0)

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

    # 霍夫圆检测：minDist=最小圆心间距，param2越小越敏感
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
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

# 色块的颜色配置
thresholds_1 = [[0, 80, 40, 80, 10, 80]]      # red
thresholds_2 = [[20, 70, -20, 5, -40, 0]]   # blue [0, 45, -35, 45, -45, -5][20, 50, -15, 5, -40, 0]
thresholds_3 = [[20, 75, -55, -15, 5, 45]]    # green
thresholds_4 = [[55, 85, -10, 10, -10, 20]]   # white[50, 75, -15, 25, -10, 15]
thresholds_5 = [[0, 25, -10, 10, -15, 5]]    # black

def find_circles_ABC(data1, data2, data3, data4):
    """打包并发送数据"""
    # 将数据项转换为字节数据，确保值在0-255范围内
    
    def clamp_byte(value):
        """确保值在0-255范围内"""
        return max(0, min(255, value))
    """打包并发送数据"""
    # 确保数据在有效范围内
    data1 = clamp_byte(data1)
    data2 = clamp_byte(data2)
    data3 = clamp_byte(data3)
    data4 = clamp_byte(data4)

    # 构建数据包（与 main.py send_packet 协议一致：单字节 d3/d4 + 2个0填充）
    packet = bytes([HEADER1, HEADER2]) + \
             data1.to_bytes(1, 'big') + data2.to_bytes(1, 'big') + \
             data3.to_bytes(1, 'big') + data4.to_bytes(1, 'big') + \
             (0).to_bytes(1, 'big') + (0).to_bytes(1, 'big') + \
             bytes([FOOTER1, FOOTER2])
    # 发送数据
    serial0.write(packet)

    # 打印发送的数据
    print(f"Sent: {[f'0x{b:02X}' for b in packet]}")
    print(f"Values: data1={data1}, data2={data2}, data3={data3}, data4={data4}")

# 主循环
#led.toggle()
while not app.need_exit():
    led.value(0)
    img = cam.read()
    # 串口接收数据（读1字节，没数据就继续等，不再强制进入任务一）
    data = serial0.read(1)
    time.sleep_ms(5)
    if data is None:
        time.sleep_ms(10)
        continue
    # 任务一：圆检测
    if data == b'\x01':
        led.value(0)
        while True:
              
            img = cam.read()
            t = time.ticks_ms()  # 获取当前时间戳（毫秒）
            # 在图像中心绘制检测区域
            center_x = img.width() // 2
            center_y = img.height() // 2
            img.draw_rect(center_x , center_y , 6, 6, image.COLOR_RED, thickness=1)
            
            # 使用改进的圆检测算法
            current_pos = detect_circle(img,30)
           
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
            
            # 发送数据
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
            print(f"FPS: {fps:.1f}, Data: {data1},{data2},{data3},{data4}")
            
            # 增加帧计数器
            frame_count += 1
    # 任务er：圆检测
    #识别冠亚
    elif data == b'\x02':
        led.value(1)
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
            
            # 发送数据
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
            print(f"FPS: {fps:.1f}, Data: {data1},{data2},{data3},{data4}")
            
            # 增加帧计数器
            frame_count += 1
    
    # 任务六：圆检测
    elif data == b'\x06':
        led.value(1)
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
            
            # 发送数据
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
            print(f"FPS: {fps:.1f}, Data: {data1},{data2},{data3},{data4}")
            
            # 增加帧计数器
            frame_count += 1
    # 任务三：色块识别
    elif data == b'\x03':
        # 初始化状态变量
        led.value(0)
        last_detected_color = None
        detection_confidence = 0
        MAX_CONFIDENCE = 3  # 连续检测到3帧才确认颜色
    
        while True:
            img = cam.read()
            img_maix = img.copy()
        
             # 定义中心区域ROI
            roi_center = (110, 90, 130, 110)  # (x, y, w, h) 调大并下移
            img_maix.draw_rect(110, 90, 130, 110, image.COLOR_WHITE)
        
            # 初始化数据
            data1 = data2 = data3 = data4 = 0
            current_color = None
        
            # 同时检测所有颜色
            red_blobs = img.find_blobs(thresholds_1, area_threshold=1000, pixels_threshold=1000, roi=roi_center)
            blue_blobs = img.find_blobs(thresholds_2, area_threshold=1000, pixels_threshold=1000, roi=roi_center)
            green_blobs = img.find_blobs(thresholds_3, area_threshold=1000, pixels_threshold=1000, roi=roi_center)
            white_blobs = img.find_blobs(thresholds_4, area_threshold=3000, pixels_threshold=3000, roi=roi_center)
            black_blobs = img.find_blobs(thresholds_5, area_threshold=4000, pixels_threshold=4000, roi=roi_center)
        
            # 选择最显著的色块（面积最大）
            all_blobs = []
            if red_blobs:
                all_blobs.extend([(blob, "Red", 21) for blob in red_blobs])
            if blue_blobs:
                all_blobs.extend([(blob, "Blue", 22) for blob in blue_blobs])
            if green_blobs:
                all_blobs.extend([(blob, "Green", 23) for blob in green_blobs])
            if white_blobs:
                all_blobs.extend([(blob, "White", 24) for blob in white_blobs])
            if black_blobs:
                all_blobs.extend([(blob, "Black", 25) for blob in black_blobs])
        
            # 如果有检测到色块
            if all_blobs:
                # 选择面积最大的色块
                largest_blob = max(all_blobs, key=lambda x: x[0].area())
                blob, color_name, color_code = largest_blob
            
                # 绘制检测结果
                img_maix.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
                img_maix.draw_string(blob[0], blob[1] - 10, color_name, image.COLOR_BLACK)
            
                # 更新颜色状态
                if color_name == last_detected_color:
                    detection_confidence = min(detection_confidence + 1, MAX_CONFIDENCE)
                else:
                    detection_confidence = 1
                    last_detected_color = color_name
            
                # 达到置信度阈值时发送数据
                if detection_confidence >= MAX_CONFIDENCE:
                    data1 = color_code
                    current_color = color_name
            else:
                # 没有检测到色块
                detection_confidence = max(0, detection_confidence - 1)
                if detection_confidence == 0 and last_detected_color:
                # 发送无颜色状态
                    data1 = 0
                    last_detected_color = None
        
            # 发送数据
            if data1 != 0:
                find_circles_ABC(data1, data2, data3, data4)
        
            # 显示图像
            disp.show(img_maix)
        
            # 显示当前状态
            if current_color:
                print(f"检测到颜色: {current_color} (置信度: {detection_confidence}/{MAX_CONFIDENCE})")
            elif last_detected_color and detection_confidence > 0:
                print(f"颜色丢失中: {last_detected_color} (置信度: {detection_confidence}/{MAX_CONFIDENCE})")
            else:
                print("未检测到颜色")
        
            # 检查是否退出任务三
            data_in = serial0.read()
            if data_in == b'\x04':
                data = data_in
                break
    # 显示图像
    disp.show(img)
    # 短暂延迟，避免CPU过载
    time.sleep_ms(10)
