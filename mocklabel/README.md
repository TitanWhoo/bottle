# 医药标签图像生成工具

这是一个用于生成医药标签图像的工具，可以生成带有随机文本、缺陷和透视变换的标签图像，用于训练OCR模型。

## 功能特点

- 生成包含医药相关文本的标签图像
- 支持自定义字体、字体大小、字体粗细
- 可调整图像尺寸、生成数量
- 随机添加文本缺陷（遮挡、模糊、擦除）
- 应用轻微透视变换和旋转
- 生成标注数据（JSON格式）
- 支持命令行参数配置

## 安装依赖

```bash
pip install pillow numpy opencv-python
```

## 使用方法

### 基本用法

```bash
python gen.py
```

这将使用默认参数生成100个标签图像，保存在`generated_labels`目录中。

### 自定义参数

```bash
python gen.py --font_path "SimSun.ttf" --output_dir "my_labels" --num_samples 50
```

### 所有可用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--font_path` | 字体文件路径 | STSong.ttf |
| `--output_dir` | 输出目录 | generated_labels |
| `--num_samples` | 生成样本数量 | 100 |
| `--min_font_size` | 最小字体大小 | 32 |
| `--max_font_size` | 最大字体大小 | 45 |
| `--min_width` | 最小图像宽度 | 400 |
| `--max_width` | 最大图像宽度 | 600 |
| `--min_height` | 最小图像高度 | 100 |
| `--max_height` | 最大图像高度 | 150 |
| `--font_weight` | 字体粗细 (normal/bold) | normal |
| `--defect_probability` | 添加缺陷的概率 (0-1) | 1.0 |
| `--perspective_probability` | 应用透视变换的概率 (0-1) | 0.7 |
| `--custom_text` | 自定义文本 | None |

## 示例

### 生成使用粗体的标签

```bash
python gen.py --font_weight bold
```

### 生成自定义文本的标签

```bash
python gen.py --custom_text "碳酸氢钠注射液 批号: 2405529 规格: 10ml:0.84g"
```

### 生成更大尺寸的标签

```bash
python gen.py --min_width 600 --max_width 800 --min_height 150 --max_height 200
```

## 输出

- 生成的图像保存在指定的输出目录中
- 标注数据保存在输出目录下的`dataset_annotations.json`文件中
- 标注数据包含图像文件名、原始文本、可见文本和字符边界框信息

## 注意事项

- 确保指定的字体文件存在
- 如果需要使用粗体，请确保有对应的粗体字体文件（通常命名为`*Bold.ttf`）
- 生成的图像质量取决于字体大小和图像尺寸的匹配度 