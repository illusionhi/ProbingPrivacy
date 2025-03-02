import json

input_file = "./llava_v1_5_mix665k_train_vg_text_new.json"
output_file = "./llava_v1_5_mix665k_train_vg_text_transfer.json"

def process_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        for conv in item["conversations"]:
            conv["value"] = conv["value"].replace("<image>\n", "")
            conv["value"] = conv["value"].replace("<image>", "")
        
        if item["conversations"]:
            item["conversations"][0]["value"] = "<image>\n" + item["conversations"][0]["value"]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_json(input_file, output_file)