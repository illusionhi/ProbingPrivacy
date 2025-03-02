import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_tensor_files(folder1, folder2):

    if not os.path.exists(folder1):
        print(f"Folder {folder1} not found!")
        return

    if not os.path.exists(folder2):
        print(f"Folder {folder2} not found!")
        return

    hidden_states = []
    labels = []

    def load_tensors_from_folder(folder, label):
        for filename in os.listdir(folder):
            if filename.endswith(".pt"):
                file_path = os.path.join(folder, filename)
                tensor = torch.load(file_path)
                tensor = tensor.flatten()
                hidden_states.append(tensor.float().cpu().numpy())
                labels.append(label)

    load_tensors_from_folder(folder1, r'Embedded with $\mathcal{U}_1$')
    load_tensors_from_folder(folder2, r'Embedded with $\mathcal{U}_2$')

    hidden_states = np.array(hidden_states)
    print(f"Extracted hidden states shape: {hidden_states.shape}")

    # 将标签映射为数值
    label_mapping = {
        r'Embedded with $\mathcal{U}_1$': 0,
        r'Embedded with $\mathcal{U}_2$': 1
    }
    labels_numeric = [label_mapping[label] for label in labels]

    pca = PCA(n_components=100)
    pca_result = pca.fit_transform(hidden_states)
    print(f"Shape after PCA: {pca_result.shape}")

    # 再用 t-SNE 将 PCA 后的数据降到 2 维
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42)
    tsne_result = tsne.fit_transform(pca_result)
    print(f"Shape after t-SNE: {tsne_result.shape}")

    # 设置绘图风格
    sns.set_theme(context='paper', style='whitegrid', palette='deep', font='Arial', font_scale=1.2)

    fig_tsne, ax_tsne = plt.subplots(figsize=(7, 5))
    scatter_tsne = ax_tsne.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=labels_numeric,
        cmap='viridis',
        s=30,
        alpha=0.8
    )
    ax_tsne.set_title('Probing Accuracy:%', fontsize=16)
    ax_tsne.legend(
        handles=scatter_tsne.legend_elements()[0], 
        labels=[r'Embedded with $\mathcal{U}_1$', r'Embedded with $\mathcal{U}_2$'],
        loc='best', 
        fontsize=16
    )
    plt.tight_layout()
    plt.savefig('vg_r100_12.png')
    # plt.savefig('visualization_privacy_100_username.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig_tsne)

if __name__ == "__main__":
    folder1 = "./Qwen-VL/results/qwen_userid/test1_origin"
    folder2 = "./Qwen-VL/results/qwen_userid/test2_origin"
    visualize_tensor_files(folder1, folder2)