import json
import tqdm
import transformers
import torch

# ========== 1) 初始化 Llama 推理管线 ==========

# 替换为你需要的 Meta-Llama-3-8B-Instruct 的模型 ID
model_id = "./Llama-3.1-8B-Instruct"

# 构建一个 text-generation 的pipeline
# 注意根据环境选择合适的 device_map、数据类型等
paraphrase_pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)
paraphrase_pipeline.model.generation_config.pad_token_id = paraphrase_pipeline.tokenizer.pad_token_id
# 准备一下终止 token (可选)
terminators = [
    paraphrase_pipeline.tokenizer.eos_token_id,
    # 若模型中定义了特殊的 <|eot_id|>，也可加上
    paraphrase_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


def paraphrase_text_with_llama(text: str) -> str:
    """
    使用 Meta-Llama-3-8B-Instruct 对文本进行同义改写，要求:
    - 只改变 1~2 个词，尽可能少地改变原句
    - 保持原本语义不变
    - 若文本包含<image>，则必须保留<image>原样不变，不要删除或修改它
    """

    # 1) 先拆分 <image> 部分（如果存在）
    if text.startswith("<image>"):
        # 假设格式固定为 "<image>\n" 然后才是正文
        split_parts = text.split("\n", 1)
        if len(split_parts) == 2:
            image_part = split_parts[0] + "\n"  # "<image>\n"
            rest_part = split_parts[1]
        else:
            # 即使没有换行，也至少把<image>先提取出来
            image_part = "<image>"
            rest_part = text[len("<image>"):]
    else:
        # 如果文本不含<image>或不以<image>开头，则整体作为rest_part
        image_part = ""
        rest_part = text

    # 如果 rest_part 为空(例如只有<image>没有其它文本)，则无需改写
    if not rest_part.strip():
        return text

    # 2) 构造提示，指导 Llama 做同义改写
    system_prompt = (
        "You are a helpful assistant that carefully modifies text while preserving the original meaning. "
        "You will only replace or slightly alter one or two words or punctuation with synonyms, ensuring minimal change. "
        "Do not alter the text structure or meaning beyond this. "
        "If the text starts with <image>, keep that part exactly as is and do not remove or alter <image> in any way."
    )
    user_prompt = (
        f"Original text:\n{rest_part}\n\n"
        "Rewrite it by changing only one or two words or punctuation to synonyms without any other words. "
        "Do not add any unrelated content."
    )

    # 3) 调用 Llama 进行推理
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        outputs = paraphrase_pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,  # 使用上面定义的终止 token
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        paraphrased = outputs[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        print("Llama paraphrase pipeline call failed:", e)
        paraphrased = rest_part

    # 4) 组合 <image> 和改写后的文本
    return image_part + paraphrased


def main():
    # ========== 2) 读入原始 JSON 数据 ==========

    input_filename = "newnewnew.json"
    output_filename = "newnewnew222.json"

    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    # ========== 3) 对每条数据的对话内容做同义改写 ==========

    for item in tqdm.tqdm(data[:]):  # 示例：只处理第 4000~7000 条
        if "conversations" in item:
            for conv in item["conversations"]:
                original_text = conv["value"]
                # 调用使用 Llama 的改写函数
                new_text = paraphrase_text_with_llama(original_text)
                conv["value"] = new_text
        new_data.append(item)

    # ========== 4) 写回改写后的结果到新的 JSON 文件 ==========

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()