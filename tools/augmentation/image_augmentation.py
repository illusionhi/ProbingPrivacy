import os
import random
import shutil
from PIL import Image, ImageEnhance, UnidentifiedImageError

def augment_image(img):
    """
    对输入的PIL图像做一系列的随机数据增强，并返回增强后的图像。
    包含：随机旋转、随机水平翻转、随机亮度调整、随机对比度调整。
    """
    
    # 1. 随机旋转（-30° ~ +30°）
    angle = random.randint(-30, 30)
    img = img.rotate(angle, expand=True)
    
    # 2. 随机水平翻转
    # 以 50% 的概率进行水平翻转
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 3. 随机亮度调整（范围 [0.8, 1.2]）
    brightness_factor = random.uniform(0.8, 1.2)
    enhancer_bri = ImageEnhance.Brightness(img)
    img = enhancer_bri.enhance(brightness_factor)
    
    # 4. 随机对比度调整（范围 [0.8, 1.2]）
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer_con = ImageEnhance.Contrast(img)
    img = enhancer_con.enhance(contrast_factor)

    return img

def batch_augment_images(input_folder, output_folder):
    """
    批量对 input_folder 中的图片进行数据增强，并将结果保存到 output_folder。
    """
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 列举输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理常见的图像格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 打开图像
            try:
                # 打开图像
                with Image.open(input_path) as img:
                    # 转换为RGB保证一致性
                    img = img.convert('RGB')
                    
                    # 数据增强
                    aug_img = augment_image(img)
                    
                    # 保存结果
                    aug_img.save(output_path)
            
            except UnidentifiedImageError:
                # 如果图像无法识别，跳过并打印错误信息
                shutil.copy(input_path, output_path)
                print(f"无法识别的图片已复制到 output_folder: {filename}")

if __name__ == "__main__":
    # 示例：自定义输入、输出文件夹路径
    input_folder = "./data_without_privacy/train/vg/VG_100K_2"   # 替换为实际的输入文件夹路径
    output_folder = "./data_without_privacy/train_image_augmentation/vg/VG_100K_2" # 替换为实际的输出文件夹路径

    batch_augment_images(input_folder, output_folder) 