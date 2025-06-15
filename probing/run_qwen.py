import argparse
import torch
import os
from glob import glob
import sys
from peft import PeftModel
sys.path.append('/data1/jutj/.cache')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 使用第一个GPU，修改为你想使用的卡号

# 模型保存路径

# 导入transformers库
from transformers import AutoModelForCausalLM, AutoTokenizer

from PIL import Image
import requests
from io import BytesIO
import re

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def get_layer_output_hook(layer_index):
    def hook(module, input, output):
        global layer_outputs
        # 只保存每层的最后一个token的输出
        if isinstance(output, tuple):
            output = output[0]
        # 确保输出是一个张量
        layer_outputs[layer_index].append(output.detach())
    return hook

def eval_single_image(model, tokenizer, args, image_file, test_type, test_prefix):
    global layer_outputs
    # 每层初始化一个空列表
    layer_outputs = {i: [] for i in range(len(model.transformer.h))}

    hook_handles = []

    # 为每一层注册钩子
    for i, layer in enumerate(model.transformer.h):
        hook_handle = layer.register_forward_hook(get_layer_output_hook(i))
        hook_handles.append(hook_handle)

    # 载入图片
    image = load_image(image_file)

    # 使用本地微调的Qwen模型进行推理
    with torch.no_grad():
        response, history = model.chat(tokenizer, query=f'<img>{image_file}</img>{args.query}', history=None, max_new_tokens=args.max_new_tokens)

    # 打印推理结果
    print(f"Query: {args.query}\nImage: {image_file}\nResponse: {response}")

    # 设置输出文件路径
    output_file = f"probing/results/username_{test_type + test_prefix}.txt"

    # 保存所有层的最后一个token的输出到同一个文件
    if layer_outputs:
        with open(output_file, "a") as f:
            f.write(f"Image: {image_file}\n")  # 记录图片文件名

            # 遍历每层，保存每层最后一个token的向量数据
            for layer_idx, outputs in layer_outputs.items():
                if outputs:  # 确保有输出
                    last_layer_output = outputs[-1]
                    last_token_output = last_layer_output[:, -2, :].to(torch.float32).cpu().detach().numpy().tolist()
                    f.write(f"Last token vector from layer {layer_idx}: {last_token_output}\n")

            f.write("=" * 50 + "\n")  # 添加分隔符
        print(f"All layers' last token outputs saved to '{output_file}'.")

    # 移除所有钩子
    for hook_handle in hook_handles:
        hook_handle.remove()

def eval_folder_of_images(args, image_folder, test_type, test_prefix):
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_base, device_map="cuda", trust_remote_code=True).eval()
    if "_r50" in test_prefix:
        model = PeftModel.from_pretrained(model, "/data1/jutj/Qwen-VL/master/checkpoints/gqa/r50/")
    elif "_r100" in test_prefix:
        model = PeftModel.from_pretrained(model, "/data1/jutj/Qwen-VL/master/checkpoints/gqa/r100/")

    print(model)

    # 获取文件夹中的所有图像文件
    image_files = glob(os.path.join(image_folder, "*.jpg"))
    if not image_files:
        print(f"No JPG images found in folder {image_folder}")
        return

    for image_file in image_files[:500]:
        eval_single_image(model, tokenizer, args, image_file, test_type, test_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen-VL-Chat")  # 预训练模型路径
    # parser.add_argument("--query", type=str, default="What is the user_id of the username?")
    parser.add_argument("--query", type=str, default="What is the username?")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1)

    args = parser.parse_args()

    test_type_list = ["test1", "test2"]
    test_prefix_list = ["_r100", "_r50", "_origin"]
    # test_prefix_list = ["_r50"]
    for test_type in test_type_list:
        for test_prefix in test_prefix_list:
            layer_outputs = []
            image_folder = f"/data1/jutj/Qwen-VL/data_test/{test_type}/gqa/useful"

            eval_folder_of_images(args, image_folder, test_type, test_prefix)
