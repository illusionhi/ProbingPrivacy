import os
import shutil
import random

def split_images(source_dir, target_dir_1, target_dir_2, split_ratio=0.8):
    # 获取 source_dir 中所有的 jpg 文件
    jpg_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
    
    # 打乱文件顺序
    random.shuffle(jpg_files)
    
    # 计算每个新文件夹应该包含的文件数量
    split_point = int(len(jpg_files) * split_ratio)
    
    # 创建目标文件夹，如果没有的话
    os.makedirs(target_dir_1, exist_ok=True)
    os.makedirs(target_dir_2, exist_ok=True)
    
    # 将文件分配到两个文件夹中
    for i, file in enumerate(jpg_files):
        src_file = os.path.join(source_dir, file)
        if i < split_point:
            dst_file = os.path.join(target_dir_1, file)
        else:
            dst_file = os.path.join(target_dir_2, file)
        
        # 复制文件到目标文件夹
        shutil.copy(src_file, dst_file)

    print(f"已将文件随机划分并复制到 {target_dir_1}（{int(split_ratio*100)}%）和 {target_dir_2}（{int((1-split_ratio)*100)}%）中。")

# 调用函数，示例：
# source_directory = 'hy/data/coco_hy/split2'  # 源文件夹路径
# target_directory_1 = 'hy/data/coco_hy/split2_train'  # 目标文件夹1路径
# target_directory_2 = 'hy/data/coco_hy/split2_test'  # 目标文件夹2路径

source_directory = '/data1/jutj/LLaVA/playground/test/test1/gqa/images'  # 源文件夹路径
target_directory_1 = '/data1/jutj/LLaVA/playground/test0/test1/gqa/useless'  # 目标文件夹1路径
target_directory_2 = '/data1/jutj/LLaVA/playground/test0/test1/gqa/useful'  # 目标文件夹2路径

split_images(source_directory, target_directory_1, target_directory_2)
