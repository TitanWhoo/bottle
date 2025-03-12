from PIL import Image, ImageDraw, ImageFont, ImageFont
import numpy as np
import cv2
import random
import json
import os
import argparse
from typing import Tuple, List, Dict, Any, Optional

class LabelGenerator:
    """医药标签图像生成器"""
    
    # 医药相关文本样本 - 固定内容列表
    MEDICINE_NAMES = [
        "碳酸氢钠注射液",
        "注射用青霉素钠",
        "氯化钠注射液",
        "葡萄糖注射液",
        "5%葡萄糖注射液",
        "0.9%氯化钠注射液",
        "盐酸利多卡因注射液",
        "维生素C注射液",
        "甲硝唑注射液",
        "复方氨基酸注射液"
    ]

    MANUFACTURER_NAMES = [
        "许昌格锐特",
        "国药集团",
        "华润医药",
        "齐鲁制药",
        "恒瑞医药",
        "扬子江药业",
        "石药集团",
        "天士力制药"
    ]

    SPECIFICATION_TEMPLATES = [
        "规格: {}ml:{}g",
        "规格: {}ml",
        "规格: {}mg/支",
        "规格: {}g/瓶"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化标签生成器
        
        Args:
            config: 配置参数字典
        """
        self.font_path = config.get('font_path', "STSong.ttf")
        self.output_dir = config.get('output_dir', "output")
        self.num_samples = config.get('num_samples', 100)
        self.min_font_size = config.get('min_font_size', 32)
        self.max_font_size = config.get('max_font_size', 45)
        self.min_width = config.get('min_width', 400)
        self.max_width = config.get('max_width', 600)
        self.min_height = config.get('min_height', 100)
        self.max_height = config.get('max_height', 150)
        self.font_weight = config.get('font_weight', 'normal')  # 'normal' 或 'bold'
        self.defect_probability = config.get('defect_probability', 1.0)  # 添加缺陷的概率
        self.perspective_probability = config.get('perspective_probability', 0.7)  # 应用透视变换的概率
        self.custom_text = config.get('custom_text', None)  # 自定义文本
        self.label_type = config.get('label_type', 'random')  # 标签类型
        self.medicine_file = config.get('medicine_file', "medicine.txt")  # 药品名称文件
        
        # 从文件加载药品名称
        self.medicine_names = self._load_medicine_names()
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _load_medicine_names(self) -> List[str]:
        """
        从文件加载药品名称
        
        Returns:
            药品名称列表
        """
        # 如果文件不存在，使用默认列表
        if not os.path.exists(self.medicine_file):
            print(f"警告：药品名称文件 {self.medicine_file} 不存在，使用默认列表")
            return self.MEDICINE_NAMES
        
        try:
            with open(self.medicine_file, 'r', encoding='utf-8') as f:
                # 读取非空行
                names = [line.strip() for line in f if line.strip()]
            
            if not names:
                print(f"警告：药品名称文件 {self.medicine_file} 为空，使用默认列表")
                return self.MEDICINE_NAMES
                
            print(f"已从 {self.medicine_file} 加载 {len(names)} 个药品名称")
            return names
        except Exception as e:
            print(f"读取药品名称文件时出错：{e}，使用默认列表")
            return self.MEDICINE_NAMES
    
    def _add_concentration_to_liquid(self, name: str) -> str:
        """
        为含有"液"字的药品名称添加随机浓度
        
        Args:
            name: 药品名称
            
        Returns:
            添加浓度后的药品名称
        """
        if "液" in name and random.random() > 0.5:  # 50%的概率添加浓度
            # 生成随机浓度
            if random.random() > 0.7:  # 30%的概率使用小数
                concentration = f"{random.uniform(0.1, 0.9):.1f}%"
            else:  # 70%的概率使用整数
                concentration = f"{random.randint(1, 20)}%"
            
            # 在药品名称前添加浓度
            parts = name.split("液", 1)
            return f"{parts[0]}{concentration}液{parts[1] if len(parts) > 1 else ''}"
        
        return name
    
    def generate_label(self, text: str, size: Tuple[int, int], font_size: int, 
                      color: Tuple[int, int, int]) -> Tuple[Optional[Image.Image], List]:
        """
        生成完整的标签图像
        
        Args:
            text: 要生成的文本
            size: 图像尺寸 (宽, 高)
            font_size: 字体大小
            color: 文本颜色 (R, G, B)
            
        Returns:
            生成的图像和字符位置信息
        """
        # 随机选择背景色（浅色）
        bg_color = (
            random.randint(235, 255),
            random.randint(235, 255),
            random.randint(235, 255)
        )
        
        img = Image.new("RGB", size, bg_color)  # 浅色背景
        draw = ImageDraw.Draw(img)

        try:
            # 根据字体粗细选择字体
            if self.font_weight == 'bold' and os.path.exists(self.font_path.replace('.ttf', 'Bold.ttf')):
                font_path = self.font_path.replace('.ttf', 'Bold.ttf')
            else:
                font_path = self.font_path
                
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"字体文件未找到，请检查路径：{self.font_path}")
            return None, None

        text_size = draw.textbbox((0, 0), text, font=font)  # 计算文本大小
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        text_position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

        # 记录每个字符的位置信息
        char_positions = []
        x_cursor = text_position[0]
        
        for char in text:
            char_box = draw.textbbox((x_cursor, text_position[1]), char, font=font)
            char_positions.append((char, char_box))  # 记录字符和其位置
            x_cursor = char_box[2]  # 更新x坐标

        # 绘制完整文本
        draw.text(text_position, text, font=font, fill=color)
        
        return img, char_positions

    def add_defects(self, image: Image.Image, char_positions: List) -> Tuple[Image.Image, List, List]:
        """
        对部分字符添加遮挡/模糊/擦除/部分缺失/磨损，使其不可识别或部分可识别
        
        Args:
            image: 原始图像
            char_positions: 字符位置信息
            
        Returns:
            处理后的图像、有效文本和有效边界框
        """
        img = np.array(image)
        h, w, _ = img.shape

        # 选择需要破坏的字符
        num_defects = random.randint(1, max(1, len(char_positions) // 3))  # 随机破坏 1~33% 的字符
        defect_chars = random.sample(char_positions, num_defects)

        # 记录未受损字符
        valid_text = []
        valid_boxes = []

        for char, bbox in char_positions:
            x1, y1, x2, y2 = bbox
            if (char, bbox) in defect_chars:
                defect_type = random.random()
                # 1. 遮挡字符
                if defect_type > 0.85:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
                # 2. 模糊字符
                elif defect_type > 0.7:
                    # 确保ROI区域有效
                    if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                        roi = img[y1:y2, x1:x2]
                        if roi.size > 0:  # 确保ROI不为空
                            roi = cv2.GaussianBlur(roi, (7, 7), 3)
                            img[y1:y2, x1:x2] = roi
                # 3. 擦除字符
                elif defect_type > 0.55:
                    if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                        img[y1:y2, x1:x2] = 255  # 变成白色背景，不留痕迹
                # 4. 部分缺失字符
                else:
                    if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                        char_width = x2 - x1
                        char_height = y2 - y1
                        
                        # 随机选择缺失部分的类型
                        partial_defect_type = random.randint(1, 11)
                        
                        # 1. 水平一半缺失（左半部分或右半部分）
                        if partial_defect_type == 1:
                            if random.random() > 0.5:  # 左半部分缺失
                                img[y1:y2, x1:x1+char_width//2] = 255
                            else:  # 右半部分缺失
                                img[y1:y2, x1+char_width//2:x2] = 255
                        
                        # 2. 垂直一半缺失（上半部分或下半部分）
                        elif partial_defect_type == 2:
                            if random.random() > 0.5:  # 上半部分缺失
                                img[y1:y1+char_height//2, x1:x2] = 255
                            else:  # 下半部分缺失
                                img[y1+char_height//2:y2, x1:x2] = 255
                        
                        # 3. 左上角缺失
                        elif partial_defect_type == 3:
                            corner_width = random.randint(char_width//4, char_width//2)
                            corner_height = random.randint(char_height//4, char_height//2)
                            img[y1:y1+corner_height, x1:x1+corner_width] = 255
                        
                        # 4. 右上角缺失
                        elif partial_defect_type == 4:
                            corner_width = random.randint(char_width//4, char_width//2)
                            corner_height = random.randint(char_height//4, char_height//2)
                            img[y1:y1+corner_height, x2-corner_width:x2] = 255
                        
                        # 5. 左下角缺失
                        elif partial_defect_type == 5:
                            corner_width = random.randint(char_width//4, char_width//2)
                            corner_height = random.randint(char_height//4, char_height//2)
                            img[y2-corner_height:y2, x1:x1+corner_width] = 255
                        
                        # 6. 右下角缺失
                        elif partial_defect_type == 6:
                            corner_width = random.randint(char_width//4, char_width//2)
                            corner_height = random.randint(char_height//4, char_height//2)
                            img[y2-corner_height:y2, x2-corner_width:x2] = 255
                            
                        # 7. 随机椭圆形缺失
                        elif partial_defect_type == 7:
                            # 创建一个与字符大小相同的掩码
                            mask = np.ones((char_height, char_width), dtype=np.uint8) * 255
                            # 随机椭圆中心
                            center_x = random.randint(char_width//4, 3*char_width//4)
                            center_y = random.randint(char_height//4, 3*char_height//4)
                            # 随机椭圆轴长
                            axis_x = random.randint(char_width//4, char_width//2)
                            axis_y = random.randint(char_height//4, char_height//2)
                            # 绘制椭圆
                            cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 0, -1)
                            # 应用掩码
                            roi = img[y1:y2, x1:x2].copy()
                            roi[mask == 0] = 255
                            img[y1:y2, x1:x2] = roi
                            
                        # 8. 随机多边形缺失
                        elif partial_defect_type == 8:
                            # 创建一个与字符大小相同的掩码
                            mask = np.ones((char_height, char_width), dtype=np.uint8) * 255
                            # 生成3-6个随机点
                            num_points = random.randint(3, 6)
                            points = []
                            for _ in range(num_points):
                                px = random.randint(0, char_width-1)
                                py = random.randint(0, char_height-1)
                                points.append([px, py])
                            points = np.array(points, np.int32)
                            points = points.reshape((-1, 1, 2))
                            # 绘制多边形
                            cv2.fillPoly(mask, [points], 0)
                            # 应用掩码
                            roi = img[y1:y2, x1:x2].copy()
                            roi[mask == 0] = 255
                            img[y1:y2, x1:x2] = roi
                            
                        # 9. 渐变缺失（从一侧逐渐消失）
                        elif partial_defect_type == 9:
                            # 选择渐变方向
                            gradient_direction = random.randint(1, 4)
                            roi = img[y1:y2, x1:x2].copy()
                            
                            if gradient_direction == 1:  # 从左到右渐变
                                for i in range(char_width):
                                    # 计算透明度，从左到右逐渐增加
                                    alpha = i / char_width
                                    if random.random() > alpha:
                                        roi[:, i] = 255
                                        
                            elif gradient_direction == 2:  # 从右到左渐变
                                for i in range(char_width):
                                    # 计算透明度，从右到左逐渐增加
                                    alpha = (char_width - i) / char_width
                                    if random.random() > alpha:
                                        roi[:, i] = 255
                                        
                            elif gradient_direction == 3:  # 从上到下渐变
                                for i in range(char_height):
                                    # 计算透明度，从上到下逐渐增加
                                    alpha = i / char_height
                                    if random.random() > alpha:
                                        roi[i, :] = 255
                                        
                            else:  # 从下到上渐变
                                for i in range(char_height):
                                    # 计算透明度，从下到上逐渐增加
                                    alpha = (char_height - i) / char_height
                                    if random.random() > alpha:
                                        roi[i, :] = 255
                                        
                            img[y1:y2, x1:x2] = roi
                            
                        # 10. 笔画缺失效果
                        elif partial_defect_type == 10:
                            roi = img[y1:y2, x1:x2].copy()
                            
                            # 将ROI转换为灰度图
                            if roi.shape[2] == 3:  # 确保是彩色图像
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            else:
                                roi_gray = roi[:,:,0]  # 如果已经是灰度图，直接取第一个通道
                            
                            # 二值化处理，提取字符轮廓
                            _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            
                            # 查找轮廓
                            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                # 随机选择要擦除的笔画数量
                                num_strokes_to_erase = random.randint(1, min(3, len(contours)))
                                strokes_to_erase = random.sample(contours, num_strokes_to_erase)
                                
                                # 创建掩码
                                mask = np.zeros_like(binary)
                                
                                # 在掩码上绘制要擦除的笔画
                                for contour in strokes_to_erase:
                                    cv2.drawContours(mask, [contour], -1, 255, -1)
                                
                                # 应用掩码，擦除选定的笔画
                                roi[mask == 255] = 255
                                
                                # 如果没有找到足够的轮廓，随机绘制线条模拟笔画缺失
                                if len(contours) < 2:
                                    # 随机绘制1-3条线
                                    for _ in range(random.randint(1, 3)):
                                        start_x = random.randint(0, char_width-1)
                                        start_y = random.randint(0, char_height-1)
                                        end_x = random.randint(0, char_width-1)
                                        end_y = random.randint(0, char_height-1)
                                        thickness = random.randint(1, 3)
                                        cv2.line(roi, (start_x, start_y), (end_x, end_y), (255, 255, 255), thickness)
                            else:
                                # 如果没有找到轮廓，随机绘制线条模拟笔画缺失
                                for _ in range(random.randint(1, 3)):
                                    start_x = random.randint(0, char_width-1)
                                    start_y = random.randint(0, char_height-1)
                                    end_x = random.randint(0, char_width-1)
                                    end_y = random.randint(0, char_height-1)
                                    thickness = random.randint(1, 3)
                                    cv2.line(roi, (start_x, start_y), (end_x, end_y), (255, 255, 255), thickness)
                            
                            img[y1:y2, x1:x2] = roi
                            
                        # 11. 磨损效果
                        else:
                            roi = img[y1:y2, x1:x2].copy()
                            
                            # 将ROI转换为灰度图
                            if roi.shape[2] == 3:  # 确保是彩色图像
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            else:
                                roi_gray = roi[:,:,0]  # 如果已经是灰度图，直接取第一个通道
                            
                            # 二值化处理，提取字符
                            _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            
                            # 创建磨损效果掩码
                            wear_mask = np.zeros_like(binary)
                            
                            # 磨损类型
                            wear_type = random.randint(1, 3)
                            
                            # 1. 边缘磨损 - 字符边缘变得不规则
                            if wear_type == 1:
                                # 膨胀和腐蚀操作，使边缘不规则
                                kernel = np.ones((2, 2), np.uint8)
                                dilated = cv2.dilate(binary, kernel, iterations=1)
                                eroded = cv2.erode(binary, kernel, iterations=1)
                                
                                # 计算边缘区域
                                edge = cv2.subtract(dilated, eroded)
                                
                                # 随机删除边缘的一部分
                                for y in range(edge.shape[0]):
                                    for x in range(edge.shape[1]):
                                        if edge[y, x] > 0 and random.random() > 0.5:
                                            wear_mask[y, x] = 255
                            
                            # 2. 随机噪点磨损 - 在字符上添加随机白色噪点
                            elif wear_type == 2:
                                # 在字符区域内添加随机噪点
                                for y in range(binary.shape[0]):
                                    for x in range(binary.shape[1]):
                                        if binary[y, x] > 0 and random.random() > 0.85:
                                            wear_mask[y, x] = 255
                            
                            # 3. 纹理磨损 - 模拟划痕或磨损纹理
                            else:
                                # 生成随机线条模拟划痕
                                num_scratches = random.randint(3, 8)
                                for _ in range(num_scratches):
                                    # 随机起点和终点
                                    start_x = random.randint(0, char_width-1)
                                    start_y = random.randint(0, char_height-1)
                                    
                                    # 控制划痕长度
                                    length = random.randint(char_width//4, char_width//2)
                                    angle = random.uniform(0, 2*np.pi)
                                    
                                    end_x = int(start_x + length * np.cos(angle))
                                    end_y = int(start_y + length * np.sin(angle))
                                    
                                    # 限制终点在ROI内
                                    end_x = max(0, min(end_x, char_width-1))
                                    end_y = max(0, min(end_y, char_height-1))
                                    
                                    # 绘制划痕
                                    thickness = random.randint(1, 2)
                                    cv2.line(wear_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
                            
                            # 应用磨损效果
                            roi[wear_mask > 0] = 255
                            
                            # 额外添加一些灰色像素，模拟不完全擦除
                            gray_mask = np.zeros_like(binary)
                            for y in range(binary.shape[0]):
                                for x in range(binary.shape[1]):
                                    if binary[y, x] > 0 and random.random() > 0.9:
                                        gray_mask[y, x] = 255
                            
                            # 将一些像素变为灰色而不是完全白色
                            gray_value = random.randint(200, 240)
                            if roi.shape[2] == 3:  # 彩色图像
                                roi[gray_mask > 0] = [gray_value, gray_value, gray_value]
                            else:  # 灰度图像
                                roi[gray_mask > 0] = gray_value
                            
                            img[y1:y2, x1:x2] = roi
            else:
                valid_text.append(char)
                valid_boxes.append(bbox)

        # 添加全局噪点和模糊效果
        if random.random() > 0.3:
            # 添加噪点
            noise = np.zeros(img.shape, np.uint8)
            cv2.randu(noise, 0, 20)
            img = cv2.add(img, noise)
        
        if random.random() > 0.5:
            # 轻微模糊整个图像
            img = cv2.GaussianBlur(img, (3, 3), 0.8)
        
        # 调整对比度和亮度
        alpha = random.uniform(0.8, 1.2)  # 对比度
        beta = random.randint(-10, 10)    # 亮度
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return Image.fromarray(img), valid_text, valid_boxes

    def apply_perspective(self, image: Image.Image) -> Image.Image:
        """
        应用轻微的透视变换和旋转
        
        Args:
            image: 原始图像
            
        Returns:
            处理后的图像
        """
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # 随机旋转角度（很小的角度）
        angle = random.uniform(-2, 2)
        
        # 旋转矩阵
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # 透视变换参数（轻微变形）
        pts1 = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])
        
        # 轻微扭曲四个角点
        offset = w * 0.05  # 最大偏移量为宽度的2%
        pts2 = np.float32([
            [random.uniform(0, offset), random.uniform(0, offset)],
            [w - random.uniform(0, offset), random.uniform(0, offset)],
            [random.uniform(0, offset), h - random.uniform(0, offset)],
            [w - random.uniform(0, offset), h - random.uniform(0, offset)]
        ])
        
        # 计算透视变换矩阵
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 应用透视变换
        warped = cv2.warpPerspective(rotated, M_persp, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return Image.fromarray(warped)

    @staticmethod
    def generate_random_batch_number() -> str:
        """生成随机批号"""
        year = random.randint(22, 24)
        month = random.randint(1, 12)
        batch = random.randint(1000, 9999)
        return f"{year}{month:02d}{batch}"

    @staticmethod
    def generate_random_production_date() -> str:
        """生成随机生产日期"""
        year = random.randint(2022, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return f"{year}-{month:02d}-{day:02d}"

    @staticmethod
    def generate_random_expiry_date(prod_date: str) -> str:
        """根据生产日期生成有效期"""
        # 从生产日期解析年月日
        year, month, day = map(int, prod_date.split('-'))
        # 有效期通常为2-3年
        expiry_year = year + random.randint(2, 3)
        return f"{expiry_year}.{month:02d}."

    def generate_random_specification(self) -> str:
        """生成随机规格信息"""
        template = random.choice(self.SPECIFICATION_TEMPLATES)
        if "ml:{}g" in template:
            return template.format(random.choice([5, 10, 20]), random.choice(["0.84", "1.2", "2.5"]))
        elif "ml" in template:
            return template.format(random.choice([2, 5, 10, 20, 50, 100]))
        elif "mg/支" in template:
            return template.format(random.choice([50, 100, 250, 500]))
        else:  # g/瓶
            return template.format(random.choice(["0.5", "1", "2.5"]))

    @staticmethod
    def generate_random_approval_number() -> str:
        """生成随机国药准字编号"""
        prefix = random.choice(["H", "S", "Z"])
        numbers = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"国药准字{prefix}{numbers}"

    @staticmethod
    def generate_random_speed() -> str:
        """生成随机速度信息"""
        ml = random.choice(["1-2", "2-5", "5-10"])
        speed = random.randint(5, 10)
        return f"{ml}ml {speed}/{speed+0.5}万支/h"
    
    def generate_random_text(self) -> str:
        """生成随机组合的文本"""
        # 随机选择药品名称并可能添加浓度
        medicine_name = random.choice(self.medicine_names)
        medicine = self._add_concentration_to_liquid(medicine_name)
        
        # 随机生成其他信息
        batch_number = f"批号: {self.generate_random_batch_number()}"
        prod_date = self.generate_random_production_date()
        production_date = f"生产日期： {prod_date}"
        expiry_date = f"有效期至：{self.generate_random_expiry_date(prod_date)}"
        specification = self.generate_random_specification()
        approval_number = self.generate_random_approval_number()
        manufacturer = random.choice(self.MANUFACTURER_NAMES)
        speed_info = self.generate_random_speed()
        
        # 所有可能的标签类型
        label_types = {
            'medicine': medicine,  # 药品名称
            'manufacturer': manufacturer,  # 厂商名称
            'batch': batch_number,  # 批号
            'production_date': production_date,  # 生产日期
            'expiry_date': expiry_date,  # 有效期
            'specification': specification,  # 规格
            'approval': approval_number,  # 批准文号
            'speed': speed_info  # 速度信息
        }
        
        # 根据指定的标签类型返回对应文本
        if self.label_type != 'random' and self.label_type in label_types:
            return label_types[self.label_type]
        
        # 随机选择一种标签类型
        return random.choice(list(label_types.values()))
    
    def generate_samples(self) -> None:
        """生成样本图像和标注数据"""
        dataset = []
        
        # 显示生成任务信息
        if self.label_type != 'random':
            print(f"正在生成 {self.num_samples} 个 '{self.label_type}' 类型的标签图像...")
        else:
            print(f"正在生成 {self.num_samples} 个随机类型的标签图像...")
        
        for i in range(self.num_samples):
            # 使用自定义文本或生成随机文本
            if self.custom_text:
                label_text = self.custom_text
            else:
                label_text = self.generate_random_text()
            
            # 随机字体大小
            font_size = random.randint(self.min_font_size, self.max_font_size)
            
            # 随机蓝色色调
            blue_color = (
                random.randint(0, 30),
                random.randint(0, 30),
                random.randint(180, 230)
            )
            
            # 随机图像尺寸
            width = random.randint(self.min_width, self.max_width)
            height = random.randint(self.min_height, self.max_height)
            
            # 生成标签
            label, char_positions = self.generate_label(
                label_text, 
                size=(width, height),
                font_size=font_size,
                color=blue_color
            )
            
            if label:
                # 添加缺陷
                if random.random() < self.defect_probability:
                    label_defect, valid_text, valid_boxes = self.add_defects(label, char_positions)
                else:
                    label_defect = label
                    valid_text = [char for char, _ in char_positions]
                    valid_boxes = [bbox for _, bbox in char_positions]
                
                # 应用透视变换和旋转
                if random.random() < self.perspective_probability:
                    label_defect = self.apply_perspective(label_defect)
                
                # 保存图像
                output_path = os.path.join(self.output_dir, f"fake_{i}.png")
                label_defect.save(output_path)

                # 生成标注数据
                annotation = {
                    "image": f"fake_{i}.png",
                    "text": "".join(valid_text),  # 仅保留未损坏的字符
                    "bboxes": valid_boxes,  # 仅保留可见字符的坐标
                    "original_text": label_text,  # 保存原始文本用于参考
                    "label_type": self.label_type if self.label_type != 'random' else self._detect_label_type(label_text)  # 记录标签类型
                }
                dataset.append(annotation)
                
                # 打印进度
                if (i + 1) % 10 == 0 or i == 0 or i == self.num_samples - 1:
                    print(f"已生成 {i + 1}/{self.num_samples} 个标签图像...")

        # 保存 JSON 标注文件
        with open(os.path.join(self.output_dir, "annotations.json"), "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
            
        # 保存 label.txt 文件
        with open(os.path.join(self.output_dir, "label.txt"), "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(f"{item['image']}\t{item['text']}\n")

        print(f"已生成 {len(dataset)} 个标签图像和标注数据！")
        print(f"输出目录: {os.path.abspath(self.output_dir)}")
        print(f"已生成 label.txt 文件，包含所有图像的文件名和标签文本")
    
    def _detect_label_type(self, text: str) -> str:
        """检测标签类型"""
        if any(name in text for name in self.medicine_names):
            return 'medicine'
        elif any(name in text for name in self.MANUFACTURER_NAMES):
            return 'manufacturer'
        elif "批号" in text:
            return 'batch'
        elif "生产日期" in text:
            return 'production_date'
        elif "有效期" in text:
            return 'expiry_date'
        elif "规格" in text:
            return 'specification'
        elif "国药准字" in text:
            return 'approval'
        elif "支/h" in text:
            return 'speed'
        else:
            return 'unknown'


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='医药标签图像生成工具')
    
    parser.add_argument('--font_path', type=str, default="Song-Light.ttf",
                        help='字体文件路径 (默认: Song-Light.ttf)')
    parser.add_argument('--output_dir', type=str, default="output",
                        help='输出目录 (默认: output)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='生成样本数量 (默认: 100)')
    parser.add_argument('--min_font_size', type=int, default=32,
                        help='最小字体大小 (默认: 32)')
    parser.add_argument('--max_font_size', type=int, default=45,
                        help='最大字体大小 (默认: 45)')
    parser.add_argument('--min_width', type=int, default=400,
                        help='最小图像宽度 (默认: 400)')
    parser.add_argument('--max_width', type=int, default=600,
                        help='最大图像宽度 (默认: 600)')
    parser.add_argument('--min_height', type=int, default=100,
                        help='最小图像高度 (默认: 100)')
    parser.add_argument('--max_height', type=int, default=150,
                        help='最大图像高度 (默认: 150)')
    parser.add_argument('--font_weight', type=str, default='normal', choices=['normal', 'bold'],
                        help='字体粗细 (默认: normal)')
    parser.add_argument('--defect_probability', type=float, default=1.0,
                        help='添加缺陷的概率 (0-1, 默认: 1.0)')
    parser.add_argument('--perspective_probability', type=float, default=0.7,
                        help='应用透视变换的概率 (0-1, 默认: 0.7)')
    parser.add_argument('--custom_text', type=str, default=None,
                        help='自定义文本 (默认: None, 使用随机生成的文本)')
    parser.add_argument('--label_type', type=str, default='random',
                        choices=['random', 'medicine', 'manufacturer', 'batch', 
                                'production_date', 'expiry_date', 'specification', 
                                'approval', 'speed'],
                        help='标签类型 (默认: random, 随机选择一种类型)')
    parser.add_argument('--medicine_file', type=str, default="medicine.txt",
                        help='药品名称文件路径 (默认: medicine.txt)')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 将参数转换为配置字典
    config = vars(args)
    
    # 创建标签生成器并生成样本
    generator = LabelGenerator(config)
    generator.generate_samples()


if __name__ == "__main__":
    main()
