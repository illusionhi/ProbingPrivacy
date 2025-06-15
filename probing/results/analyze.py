import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 加载隐藏状态数据
def load_hidden_states_from_file(file_path):
    hidden_state_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        layer_pattern = re.compile(r'Last token vector from layer (\d+): \[\[(.*?)\]\]')
        sample_data = []
        for line in lines:
            match = layer_pattern.search(line)
            if match:
                vector_str = match.group(2)
                vector = np.array([float(x) for x in vector_str.split(',')])
                sample_data.append(vector)

            if line.strip() == "==================================================":
                if sample_data:
                    hidden_state_data.append(sample_data)
                    sample_data = []

        if sample_data:
            hidden_state_data.append(sample_data)

    return np.array(hidden_state_data)

# 生成标签（每个文件对应不同标签）
def generate_labels_from_filename(file_path):
    if 'test1' in file_path:
        return 0
    elif 'test2' in file_path:
        return 1
    elif 'test3' in file_path:
        return 2

# 加载所有隐藏状态数据和标签
def load_hidden_states_and_labels(hidden_states_files):
    all_hidden_states = []
    all_labels = []

    for file_path in hidden_states_files:
        print(f"Loading data from {file_path}...")
        hidden_states = load_hidden_states_from_file(file_path)
        labels = generate_labels_from_filename(file_path)

        all_hidden_states.append(hidden_states)
        all_labels.extend([labels] * hidden_states.shape[0])

    all_hidden_states = np.concatenate(all_hidden_states, axis=0)
    all_labels = np.array(all_labels).astype(int)
    return all_hidden_states, np.array(all_labels)

# 计算并返回三个情境的准确率列表
def compute_accuracies(file_pair_1, file_pair_2, file_pair_3):
    accuracies_1 = []
    accuracies_2 = []
    accuracies_3 = []

# 情境1
    all_hidden_states_1, all_labels_1 = load_hidden_states_and_labels(file_pair_1)
    for layer_idx in range(32):  # 假设有32层
        layer_hidden_states = all_hidden_states_1[:, layer_idx, :]
        X_train, X_test, y_train, y_test = train_test_split(
            layer_hidden_states, all_labels_1, test_size=0.2, random_state=42, stratify=all_labels_1
        )
        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = round(np.mean(y_pred == y_test), 4)
        accuracies_1.append(accuracy)

    # 情境2
    all_hidden_states_2, all_labels_2 = load_hidden_states_and_labels(file_pair_2)
    for layer_idx in range(32):
        layer_hidden_states = all_hidden_states_2[:, layer_idx, :]
        X_train, X_test, y_train, y_test = train_test_split(
            layer_hidden_states, all_labels_2, test_size=0.2, random_state=42, stratify=all_labels_2
        )
        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = round(np.mean(y_pred == y_test), 4)
        accuracies_2.append(accuracy)

    # 情境3
    all_hidden_states_3, all_labels_3 = load_hidden_states_and_labels(file_pair_3)
    for layer_idx in range(32):
        layer_hidden_states = all_hidden_states_3[:, layer_idx, :]
        X_train, X_test, y_train, y_test = train_test_split(
            layer_hidden_states, all_labels_3, test_size=0.2, random_state=42, stratify=all_labels_3
        )
        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = round(np.mean(y_pred == y_test), 4)
        accuracies_3.append(accuracy)

    return accuracies_1, accuracies_2, accuracies_3

def save_accuracies(filepath, accuracies_1, accuracies_2, accuracies_3):
    np.savez(filepath, accuracies_1=accuracies_1, accuracies_2=accuracies_2, accuracies_3=accuracies_3)
    print(f"Accuracies saved to {filepath}")

def load_accuracies(filepath):
    data = np.load(filepath)
    accuracies_1 = data['accuracies_1']
    accuracies_2 = data['accuracies_2']
    accuracies_3 = data['accuracies_3']
    # print(accuracies_1)
    return accuracies_1, accuracies_2, accuracies_3

def plot_accuracies(accuracies_1, accuracies_2, accuracies_3, output_image):
    sns.set_theme(context='paper', style='whitegrid', palette='deep', font='Arial', font_scale=1.2)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # plt.figure(figsize=(6, 4))
    print(accuracies_1)

    layer_indices = range(len(accuracies_1))
    sns.lineplot(
        x=layer_indices, y=accuracies_1,
        marker='o', markersize=6, linewidth=3,
        label='w/o Fine-Tuning'
    )
    sns.lineplot(
        x=layer_indices, y=accuracies_2,
        marker='o', markersize=6, linewidth=3,
        label='Fine-Tuned with Privacy Embedding Rate: 50%'
    )
    sns.lineplot(
        x=layer_indices, y=accuracies_3,
        marker='o', markersize=6, linewidth=3,
        label='Fine-Tuned with Privacy Embedding Rate: 100%'
    )

    plt.axhline(y=0.51, linestyle='--', linewidth=3, alpha=0.8, label='Prompt Baseline')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.title('Classification Accuracy Comparison by Layer')
    plt.legend(loc='best', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_image)
    # plt.show()


if __name__ == "__main__":
    # 文件路径
    file_pair_1 = ['username_test1_origin.txt', 'username_test2_origin.txt']
    file_pair_2 = ['username_test1_r50.txt', 'username_test2_r50.txt']
    file_pair_3 = ['username_test1_r100.txt', 'username_test2_r100.txt']

    # ---- 第一步：计算准确率并存储到本地文件 ----
    save_file_name = "qwen_acc_username.npz"
    accuracies_1, accuracies_2, accuracies_3 = compute_accuracies(file_pair_1, file_pair_2, file_pair_3)
    save_accuracies(save_file_name, accuracies_1, accuracies_2, accuracies_3)

    # ---- 第二步：从本地文件加载准确率，并绘图 ----
    loaded_1, loaded_2, loaded_3 = load_accuracies(save_file_name)
    plot_accuracies(loaded_1, loaded_2, loaded_3, save_file_name.split('.')[-2]+".png")