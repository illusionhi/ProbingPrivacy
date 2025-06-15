import json
import os

# 读取 JSON 文件内容
with open('converted_captions.json', 'r') as f:
    data = json.load(f)

# 获取文件夹中所有的 jpg 文件名
folder_path = 'data/coco/train2017'  # 文件夹路径，替换为你实际的路径
jpg_files = [f'coco2017/train2017/{f}' for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 获取与文件夹中的 jpg 文件名对应的 JSON 记录
filtered_data = []
for item in data:
    if 'image' not in item:
        continue  # 跳过没有 'image' 键的记录
    image_name = item['image'] # 提取文件名
    if image_name in jpg_files:
        filtered_data.append(item)

# 将筛选后的数据保存为新的 JSON 文件
with open('coco2017.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f'筛选完成，共筛选出 {len(filtered_data)} 条数据，已保存为 coco2017.json')