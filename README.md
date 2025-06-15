# ProbingPrivacy

Reproduction Code for Paper "Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models". The preprint of our paper is publicly available at [this link](https://arxiv.org/abs/2503.01208).

## 🛠️ Installation
### Dependencies

The project requires the setup of two separate environments. Here are the steps to configure each environment:

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install -r requirements_llava.txt
```

```bash
conda create -n qwen python=3.10 -y
conda activate qwen
pip install -r requirements_qwen.txt
```
### Models
The models required for our experiments are [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) and [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5). Please download these two models and place them in a custom directory for use in subsequent experiments.

### Dataset Preparation

We conduct experiments using the following datasets: [COCO](http://images.cocodataset.org/zips/train2017.zip), [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), [OCR-VQA](https://ocr-vqa.github.io/), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), and [VisualGenome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).

Please download the dataset and the [description files](), place the description files under [data/llava](https://github.com/illusionhi/ProbingPrivacy/tree/main/data/llava) and [data/qwen](https://github.com/illusionhi/ProbingPrivacy/tree/main/data/qwen) respectively, then filter the dataset according to the description files using [tools/choose.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/tools/choose.py) and Split the dataset to obtain the training set using [tools/split.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/tools/split.py):

```bash
python choose.py
```
``` bash
python split.py
```
The training set should be organized as follows:
```
data/
└── data_without_privacy/
   └──train/
      ├── coco/
      │   └── train2017/
      ├── gqa/
      │   └── images/
      ├── ocr_vqa/
      │   └── images/
      ├── textvqa/
      │   └── train_images/
      └── vg/
          ├── VG_100K/
          └── VG_100K_2/
```
In this project, we need datasets with embedded privacy. The following are the steps for generating the privacy dataset:
1. Generate private information using [tools/generate_user_info.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/tools/generate_user_info.py)：
``` bash
python generate_user_info.py
```
2. Use [tools/add_privacy_to_image.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/tools/add_privacy_to_image.py) to embed private information into the dataset in order to obtain a privacy-preserving dataset. Modify the code at line 32 to adjust the embedding rate.
``` bash
python add_privacy_to_image.py
```
Additionally, our experiments utilize datasets with text and image augmentations.Use [tools/augmentation/text_augmentation.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/tools/augmentation/text_augmentation.py) and [tools/augmentation/image_augmentation.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/tools/augmentation/image_augmentation.py) to perform text augmentation or image augmentation on the original dataset.
``` bash
python text_augmentation.py
```
``` bash
python image_augmentation.py
```


## 📘 Instructions

### Fine-tuning
For the LLaVA model, use [finetune/finetune_lora_llava.sh](https://github.com/illusionhi/ProbingPrivacy/blob/main/finetune/finetune_lora_llava.sh) for fine-tuning. Modify --data_path to use either the original description files or the augmented description files, and modify --image_folder to use the original image data, augmented image data, or privacy-embedded image data.
```bash 
bash finetune_lora_llava.sh
```
For the Qwen model, use [finetune/finetune_lora_qwenvl.sh](https://github.com/illusionhi/ProbingPrivacy/blob/main/finetune/finetune_lora_qwenvl.sh) for fine-tuning. Modify --data_path to use various datasets.

### Evaluation
To explore how task-irrelevant content might affect the finetuning process, we examine the performance of the MLLMs on standard VQA tasks before and after embedding the task-irrelevant private content.
1. ScienceQA
Under [data/eval/scienceqa](https://github.com/illusionhi/ProbingPrivacy/blob/main/data/eval/scienceqa), download images, pid_splits.json, problems.json from the data/scienceqa folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA) and [scienceqa_test_img.jsonl](https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/scienceqa/scienceqa_test_img.jsonl).

For LLaVA, Single-GPU inference and evaluate.
```bash 
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/llava/sqa.sh
```
For Qwen-VL, run [scripts/eval/qwen/evaluate_multiple_choice.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/scripts/eval/qwen/evaluate_multiple_choice.py) as follows.
```bash 
ds="scienceqa_test_img"
checkpoint=/PATH/TO/CHECKPOINT
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_multiple_choice.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```
2. MME
Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).

For LLaVA, downloaded images to MME_Benchmark_release_version, put the official eval_tool and MME_Benchmark_release_version under [data/eval/MME](https://github.com/illusionhi/ProbingPrivacy/blob/main/data/eval/MME), then Single-GPU inference and evaluate.
``` bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/llava/mme.sh
```
For Qwen-VL:
Rearrange images by executing python [get_images.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/scripts/eval/qwen/get_images.py). 
Evaluate Qwen-VL-Chat results by executing python [eval.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/scripts/eval/qwen/eval.py).

### Cosine Gradient Similarity Comparison
Modify the Transformers library, add a path for saving gradient outputs, then run the fine-tuning script.

After obtaining the results using training datasets with different privacy rates, run the gradient similarity comparison code in tools/compute_gradients.

LLaVA:
``` bash
python gradients_llava.py
```
Qwen-VL:
``` bash
puthon gradients_qwen.py
```

### Probing
After fine-tuning, divide the test set according to the requirements in the paper, and then follow the steps below to complete the probing experiments and line chart plotting for Qwen-VL.

1. Use [probing/run_qwen.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/probing/run_qwen.py) to generate result files in probing/results.
``` bash
python run_qwen.py --model-base --query
```
2. Use [probing/results/analyze.py](https://github.com/illusionhi/ProbingPrivacy/blob/main/probing/results/analyze.py) to generate line charts based on the results.
``` bash
python analyze.py
```

