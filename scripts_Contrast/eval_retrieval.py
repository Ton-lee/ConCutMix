import argparse
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import glob
from tqdm import tqdm


def load_features(feature_folder):
    """
    加载特征数据
    """
    features = []
    labels = []
    class_folders = sorted([f for f in os.listdir(feature_folder) if os.path.isdir(os.path.join(feature_folder, f))])

    print(f"Loading features from {feature_folder}")
    for class_idx, class_folder in enumerate(tqdm(class_folders, desc="Loading classes")):
        class_path = os.path.join(feature_folder, class_folder)
        npy_files = glob.glob(os.path.join(class_path, "*.npy"))

        for npy_file in npy_files:
            feature = np.load(npy_file)
            features.append(feature)
            labels.append(class_idx)

    return np.array(features), np.array(labels), class_folders


def knn_classification(train_features, train_labels, val_features, val_labels, k_values):
    """
    kNN分类和检索性能测试
    """
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(val_features, train_features)

    classification_results = {}
    retrieval_results = {}

    for k in k_values:
        print(f"Testing k={k}...")

        # 获取top-k最近邻的索引
        topk_indices = np.argsort(-similarity_matrix, axis=1)[:, :k]

        # 分类准确率（多数投票）
        predicted_labels = []
        for i in range(len(topk_indices)):
            neighbor_labels = train_labels[topk_indices[i]]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            predicted_labels.append(unique[np.argmax(counts)])

        predicted_labels = np.array(predicted_labels)
        classification_accuracy = accuracy_score(val_labels, predicted_labels)
        classification_results[k] = classification_accuracy

        # 检索准确率（top-k中是否包含正确类别）
        correct_retrieval = np.zeros(len(val_labels))
        for i in range(len(val_labels)):
            neighbor_labels = train_labels[topk_indices[i]]
            if val_labels[i] in neighbor_labels:
                correct_retrieval[i] = 1

        retrieval_accuracy = np.mean(correct_retrieval)
        retrieval_results[k] = retrieval_accuracy

        print(f"k={k}: Classification Acc = {classification_accuracy:.4f}, Retrieval Acc = {retrieval_accuracy:.4f}")

    return classification_results, retrieval_results


def main():
    parser = argparse.ArgumentParser(description='kNN Classification and Retrieval Test')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., Cifar100-LT)')
    parser.add_argument('--max_k', type=int, default=10,
                        help='Maximum k value to test')
    parser.add_argument('--results_dir', type=str, default='/home/Users/dqy/Projects/ConCutMix/results',
                        help='Base results directory')

    args = parser.parse_args()

    # 设置路径
    base_dir = args.results_dir
    train_folder = os.path.join(base_dir, args.dataset, "features", "train")
    val_folder = os.path.join(base_dir, args.dataset, "features", "val")

    # 检查路径是否存在
    if not os.path.exists(train_folder):
        raise ValueError(f"Train folder not found: {train_folder}")
    if not os.path.exists(val_folder):
        raise ValueError(f"Validation folder not found: {val_folder}")

    # 加载特征
    print("Loading training features...")
    train_features, train_labels, train_classes = load_features(train_folder)
    print("Loading validation features...")
    val_features, val_labels, val_classes = load_features(val_folder)

    print(f"\nDataset: {args.dataset}")
    print(f"Training set: {len(train_features)} samples, {len(train_classes)} classes")
    print(f"Validation set: {len(val_features)} samples, {len(val_classes)} classes")

    # 检查类别一致性
    if len(train_classes) != len(val_classes):
        print("Warning: Number of classes in train and validation sets don't match!")

    # 测试不同的k值
    k_values = list(range(1, args.max_k + 1))
    print(f"\nTesting k values: {k_values}")

    # 运行kNN测试
    classification_acc, retrieval_acc = knn_classification(
        train_features, train_labels, val_features, val_labels, k_values
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("kNN Test Results")
    print("=" * 60)
    print(f"{'k':<4} {'Classification Acc':<18} {'Retrieval Acc':<15}")
    print("-" * 60)
    for k in k_values:
        print(f"{k:<4} {classification_acc[k]:<18.4f} {retrieval_acc[k]:<15.4f}")

    # 保存结果到文件
    output_file = os.path.join(base_dir, args.dataset, f"knn_results_{args.dataset}.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("kNN Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Training samples: {len(train_features)}\n")
        f.write(f"Validation samples: {len(val_features)}\n")
        f.write(f"Number of classes: {len(train_classes)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'k':<4} {'Classification Acc':<18} {'Retrieval Acc':<15}\n")
        f.write("-" * 60 + "\n")
        for k in k_values:
            f.write(f"{k:<4} {classification_acc[k]:<18.4f} {retrieval_acc[k]:<15.4f}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()