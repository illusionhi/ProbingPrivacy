import os
import json

def convert_llava_to_qwen(llava_json_path, qwen_json_path):
    """
    将 LLaVA 格式的对话数据转换为 Qwen-VL 格式，并确保只有第一次用户对话时带上图片信息。
    :param llava_json_path: LLaVA 格式的输入 JSON 文件路径
    :param qwen_json_path: 转换后 Qwen-VL 格式的输出 JSON 文件路径
    """
    
    # 读取 LLaVA 格式的 JSON 数据
    with open(llava_json_path, 'r', encoding='utf-8') as f:
        llava_data = json.load(f)

    qwen_data = []

    # 遍历 LLaVA JSON 列表中的每个对象
    for item in llava_data:
        llava_id = item.get("id", "")
        image_path = item.get("image", "")  # 获取图片路径
        image_path = os.path.join("./data_privacy/train-r_100", image_path)
        conversations = item.get("conversations", [])

        # 用于存放新的对话列表
        new_conversations = []
        
        # 标志位，用于判断是否是第一次用户发言
        image_inserted = False

        # 逐条处理对话
        for turn in conversations:
            speaker = turn.get("from", "")     # "human" or "gpt"
            text = turn.get("value", "")

            # 如果是用户发言，且是第一次发言才插入图片
            if speaker == "human":
                new_speaker = "user"
                text = text.replace("<image>", "").strip()  # 移除 LLaVA 格式中的图片标记
                
                # 如果是第一次用户发言，插入图片信息
                if not image_inserted and image_path:
                    text = f"Picture 1: <img>{image_path}</img>\n" + text
                    image_inserted = True  # 设置标志位，后续不会再插入图片
            elif speaker == "gpt":
                new_speaker = "assistant"
            else:
                new_speaker = speaker  # 其他角色不作改变

            # 将转换后的对话添加到新的对话列表中
            new_conversations.append({
                "from": new_speaker,
                "value": text
            })

        # 组装为 Qwen-VL 格式的数据
        qwen_item = {
            "id": llava_id,
            "conversations": new_conversations
        }
        qwen_data.append(qwen_item)

    # 写入转换后的数据到目标 JSON 文件
    with open(qwen_json_path, 'w', encoding='utf-8') as f:
        json.dump(qwen_data, f, ensure_ascii=False, indent=4)

    print(f"转换完成，结果已保存到 {qwen_json_path}")


if __name__ == "__main__":
    input_path = "./llava_v1_5_mix665k_train_coco.json"  # 输入的 LLaVA 格式文件
    output_path = "data/qwen_train_coco_r100.json"  # 输出的 Qwen-VL 格式文件
    convert_llava_to_qwen(input_path, output_path)