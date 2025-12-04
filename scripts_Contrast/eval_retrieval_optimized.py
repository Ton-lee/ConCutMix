import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score
import glob
from tqdm import tqdm
import concurrent.futures
import pickle
import time
import faiss
from pathlib import Path
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ParallelFeatureLoader:
    """并行特征加载器"""
    
    def __init__(self, feature_folder: str, use_cache: bool = True, 
                 max_workers: int = None, chunk_size: int = 1000):
        """
        初始化并行特征加载器
        
        Args:
            feature_folder: 特征文件夹路径
            use_cache: 是否使用缓存
            max_workers: 最大工作线程数
            chunk_size: 每个工作线程处理的文件块大小
        """
        self.feature_folder = feature_folder
        self.use_cache = use_cache
        self.max_workers = max_workers or min(mp.cpu_count(), 16)
        self.chunk_size = chunk_size
        self.cache_file = os.path.join(feature_folder, "features_cache.pkl")
        
    def _get_all_files(self) -> Tuple[List[str], List[int], List[str]]:
        """
        获取所有文件路径和标签
        
        Returns:
            Tuple: (文件路径列表, 标签列表, 类别文件夹列表)
        """
        class_folders = sorted([
            f for f in os.listdir(self.feature_folder) 
            if os.path.isdir(os.path.join(self.feature_folder, f))
        ])
        
        all_files = []
        all_labels = []
        
        for class_idx, class_folder in enumerate(class_folders):
            class_path = os.path.join(self.feature_folder, class_folder)
            npy_files = glob.glob(os.path.join(class_path, "*.npy"))
            
            all_files.extend(npy_files)
            all_labels.extend([class_idx] * len(npy_files))
        
        return all_files, all_labels, class_folders
    
    def _load_chunk(self, file_chunk: List[str]) -> np.ndarray:
        """
        加载一个文件块
        
        Args:
            file_chunk: 文件路径列表
            
        Returns:
            np.ndarray: 特征数组
        """
        features = []
        for file_path in file_chunk:
            try:
                # 使用内存映射模式，对于小文件更高效
                feature = np.load(file_path)
                features.append(feature.flatten())  # 确保是一维向量
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        return np.array(features) if features else np.array([])
    
    def _load_chunk_with_labels(self, file_chunk: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载一个带标签的文件块
        
        Args:
            file_chunk: (文件路径, 标签) 列表
            
        Returns:
            Tuple: (特征数组, 标签数组)
        """
        features = []
        labels = []
        for file_path, label in file_chunk:
            try:
                feature = np.load(file_path)
                features.append(feature.flatten())
                labels.append(label)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        return np.array(features), np.array(labels)
    
    def load_parallel(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        并行加载所有特征
        
        Returns:
            Tuple: (特征矩阵, 标签数组, 类别文件夹列表)
        """
        # 检查缓存
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                print(f"Loading cached features from {self.cache_file}")
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                print("Cache corrupted, regenerating...")
        
        # 获取所有文件
        all_files, all_labels, class_folders = self._get_all_files()
        
        if not all_files:
            raise ValueError(f"No .npy files found in {self.feature_folder}")
        
        print(f"Found {len(all_files)} .npy files in {len(class_folders)} classes")
        print(f"Using {self.max_workers} parallel workers")
        
        # 创建文件块（每个块包含文件路径和标签）
        file_label_pairs = list(zip(all_files, all_labels))
        chunks = [
            file_label_pairs[i:i + self.chunk_size] 
            for i in range(0, len(file_label_pairs), self.chunk_size)
        ]
        
        # 并行加载
        all_features_list = []
        all_labels_list = []
        
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_chunk = {
                executor.submit(self._load_chunk_with_labels, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # 收集结果
            for future in tqdm(
                concurrent.futures.as_completed(future_to_chunk), 
                total=len(chunks),
                desc="Loading feature chunks"
            ):
                try:
                    chunk_features, chunk_labels = future.result()
                    if len(chunk_features) > 0:
                        all_features_list.append(chunk_features)
                        all_labels_list.append(chunk_labels)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
        
        # 合并结果
        if not all_features_list:
            raise ValueError("No features loaded!")
        
        all_features = np.vstack(all_features_list)
        all_labels = np.concatenate(all_labels_list)
        
        load_time = time.time() - start_time
        print(f"\nLoaded {len(all_features)} features in {load_time:.2f} seconds")
        print(f"Features shape: {all_features.shape}")
        print(f"Labels shape: {all_labels.shape}")
        
        # 保存缓存
        if self.use_cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump((all_features, all_labels, class_folders), f, protocol=4)
                print(f"Cached features saved to {self.cache_file}")
            except Exception as e:
                print(f"Failed to save cache: {e}")
        
        return all_features, all_labels, class_folders


class FastKNNClassifier:
    """快速kNN分类器（使用FAISS加速）"""
    
    def __init__(self, train_features: np.ndarray, train_labels: np.ndarray, 
                 use_gpu: bool = False):
        """
        初始化快速kNN分类器
        
        Args:
            train_features: 训练特征
            train_labels: 训练标签
            use_gpu: 是否使用GPU
        """
        self.train_features = train_features.astype(np.float32)
        self.train_labels = train_labels
        
        # 归一化特征以便使用内积计算余弦相似度
        print("Normalizing features for cosine similarity...")
        self.train_norms = np.linalg.norm(self.train_features, axis=1, keepdims=True)
        self.train_features_norm = self.train_features / np.clip(self.train_norms, 1e-10, None)
        
        # 创建FAISS索引
        print("Building FAISS index...")
        self.dimension = self.train_features.shape[1]
        
        if use_gpu:
            try:
                # GPU索引
                res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                print("Using GPU acceleration")
            except:
                print("GPU not available, falling back to CPU")
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # CPU索引
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # 添加训练数据到索引
        self.index.add(self.train_features_norm)
        print(f"Index built with {len(train_features)} vectors")
    
    def search(self, query_features: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索k个最近邻
        
        Args:
            query_features: 查询特征
            k: 最近邻数量
            
        Returns:
            Tuple: (距离矩阵, 索引矩阵)
        """
        query_features = query_features.astype(np.float32)
        
        # 归一化查询特征
        query_norms = np.linalg.norm(query_features, axis=1, keepdims=True)
        query_features_norm = query_features / np.clip(query_norms, 1e-10, None)
        
        # 搜索最近邻
        distances, indices = self.index.search(query_features_norm, k)
        
        return distances, indices
    
    def classify(self, query_features: np.ndarray, query_labels: np.ndarray, 
                k_values: List[int], batch_size: int = 1000) -> Dict[int, Dict[str, float]]:
        """
        执行分类和检索测试
        
        Args:
            query_features: 查询特征
            query_labels: 查询标签
            k_values: k值列表
            batch_size: 批处理大小
            
        Returns:
            Dict: 每个k值的分类和检索准确率
        """
        max_k = max(k_values)
        results = {}
        
        print(f"Performing kNN search with max k={max_k}...")
        
        # 批量处理以节省内存
        all_indices = []
        
        for i in tqdm(range(0, len(query_features), batch_size), desc="Batch searching"):
            batch_features = query_features[i:i+batch_size]
            _, batch_indices = self.search(batch_features, max_k)
            all_indices.append(batch_indices)
        
        # 合并所有批次的索引
        all_indices = np.vstack(all_indices)
        
        # 对每个k值进行评估
        for k in k_values:
            print(f"\nEvaluating k={k}...")
            
            # 获取top-k最近邻
            topk_indices = all_indices[:, :k]
            
            # 计算分类准确率（多数投票）
            predicted_labels = self._majority_vote(topk_indices)
            classification_acc = accuracy_score(query_labels, predicted_labels)
            
            # 计算检索准确率（top-k中是否包含正确类别）
            retrieval_acc = self._retrieval_accuracy(topk_indices, query_labels)
            
            results[k] = {
                'classification_accuracy': classification_acc,
                'retrieval_accuracy': retrieval_acc
            }
            
            print(f"k={k}: Classification Acc = {classification_acc:.4f}, "
                  f"Retrieval Acc = {retrieval_acc:.4f}")
        
        return results
    
    def _majority_vote(self, neighbor_indices: np.ndarray) -> np.ndarray:
        """
        多数投票分类
        
        Args:
            neighbor_indices: 邻居索引矩阵
            
        Returns:
            np.ndarray: 预测标签
        """
        predicted_labels = []
        
        for i in range(len(neighbor_indices)):
            neighbor_labels = self.train_labels[neighbor_indices[i]]
            
            # 使用bincount加速多数投票
            counts = np.bincount(neighbor_labels)
            predicted_labels.append(np.argmax(counts))
        
        return np.array(predicted_labels)
    
    def _retrieval_accuracy(self, neighbor_indices: np.ndarray, query_labels: np.ndarray) -> float:
        """
        计算检索准确率
        
        Args:
            neighbor_indices: 邻居索引矩阵
            query_labels: 查询标签
            
        Returns:
            float: 检索准确率
        """
        correct = 0
        total = len(query_labels)
        
        # 使用向量化操作加速
        for i in range(total):
            neighbor_labels = self.train_labels[neighbor_indices[i]]
            if query_labels[i] in neighbor_labels:
                correct += 1
        
        return correct / total


class KNNPerformanceTester:
    """kNN性能测试器"""
    
    def __init__(self, results_dir: str, dataset: str, use_gpu: bool = False):
        """
        初始化性能测试器
        
        Args:
            results_dir: 结果目录
            dataset: 数据集名称
            use_gpu: 是否使用GPU
        """
        self.results_dir = results_dir
        self.dataset = dataset
        self.use_gpu = use_gpu
        
        # 设置路径
        self.train_folder = os.path.join(results_dir, dataset, "features", "train")
        self.val_folder = os.path.join(results_dir, dataset, "features", "val")
        
        # 检查路径
        if not os.path.exists(self.train_folder):
            raise ValueError(f"Train folder not found: {self.train_folder}")
        if not os.path.exists(self.val_folder):
            raise ValueError(f"Validation folder not found: {self.val_folder}")
    
    def run_test(self, max_k: int = 10, max_workers: int = None, 
                use_cache: bool = True, batch_size: int = 1000) -> Dict:
        """
        运行kNN性能测试
        
        Args:
            max_k: 最大k值
            max_workers: 并行工作线程数
            use_cache: 是否使用缓存
            batch_size: 批处理大小
            
        Returns:
            Dict: 测试结果
        """
        print(f"\n{'='*60}")
        print(f"kNN Performance Test - {self.dataset}")
        print(f"{'='*60}")
        
        # 1. 并行加载特征
        print("\n1. Loading features...")
        
        # 加载训练特征
        train_loader = ParallelFeatureLoader(
            self.train_folder, 
            use_cache=use_cache, 
            max_workers=max_workers,
            chunk_size=batch_size
        )
        
        start_time = time.time()
        train_features, train_labels, train_classes = train_loader.load_parallel()
        train_load_time = time.time() - start_time
        
        # 加载验证特征
        val_loader = ParallelFeatureLoader(
            self.val_folder, 
            use_cache=use_cache, 
            max_workers=max_workers,
            chunk_size=batch_size
        )
        
        start_time = time.time()
        val_features, val_labels, val_classes = val_loader.load_parallel()
        val_load_time = time.time() - start_time
        
        print(f"\nTraining set: {len(train_features)} samples, {len(train_classes)} classes")
        print(f"Validation set: {len(val_features)} samples, {len(val_classes)} classes")
        print(f"Feature dimension: {train_features.shape[1]}")
        print(f"Train loading time: {train_load_time:.2f}s")
        print(f"Validation loading time: {val_load_time:.2f}s")
        
        # 2. 训练kNN分类器
        print("\n2. Building kNN classifier...")
        knn_classifier = FastKNNClassifier(
            train_features, train_labels, use_gpu=self.use_gpu
        )
        
        # 3. 运行测试
        print("\n3. Running kNN tests...")
        k_values = list(range(1, max_k + 1))
        print(f"Testing k values: {k_values}")
        
        start_time = time.time()
        results = knn_classifier.classify(val_features, val_labels, k_values, batch_size=batch_size)
        test_time = time.time() - start_time
        
        print(f"\nTotal test time: {test_time:.2f} seconds")
        print(f"Average time per k: {test_time/len(k_values):.2f} seconds")
        
        # 4. 整理结果
        final_results = {
            'dataset': self.dataset,
            'train_samples': len(train_features),
            'val_samples': len(val_features),
            'num_classes': len(train_classes),
            'feature_dim': train_features.shape[1],
            'train_load_time': train_load_time,
            'val_load_time': val_load_time,
            'test_time': test_time,
            'results': results
        }
        
        # 5. 保存结果
        self._save_results(final_results)
        
        return final_results
    
    def _save_results(self, results: Dict):
        """
        保存测试结果
        
        Args:
            results: 测试结果
        """
        # 创建输出目录
        output_dir = os.path.join(self.results_dir, self.dataset, "knn_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果到文本文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"knn_results_{self.dataset}_{timestamp}.txt")
        
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("kNN PERFORMANCE TEST RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Dataset: {results['dataset']}\n")
            f.write(f"Training samples: {results['train_samples']}\n")
            f.write(f"Validation samples: {results['val_samples']}\n")
            f.write(f"Number of classes: {results['num_classes']}\n")
            f.write(f"Feature dimension: {results['feature_dim']}\n")
            f.write(f"Train loading time: {results['train_load_time']:.2f}s\n")
            f.write(f"Validation loading time: {results['val_load_time']:.2f}s\n")
            f.write(f"Total test time: {results['test_time']:.2f}s\n\n")
            
            f.write("=" * 70 + "\n")
            f.write(f"{'k':<6} {'Classification Acc':<20} {'Retrieval Acc':<15}\n")
            f.write("-" * 70 + "\n")
            
            for k, k_results in results['results'].items():
                f.write(f"{k:<6} {k_results['classification_accuracy']:<20.4f} "
                       f"{k_results['retrieval_accuracy']:<15.4f}\n")
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # 保存摘要结果到CSV文件（便于后续分析）
        csv_file = os.path.join(output_dir, f"knn_summary_{self.dataset}_{timestamp}.csv")
        
        with open(csv_file, 'w') as f:
            f.write("k,classification_accuracy,retrieval_accuracy\n")
            for k, k_results in results['results'].items():
                f.write(f"{k},{k_results['classification_accuracy']:.4f},"
                       f"{k_results['retrieval_accuracy']:.4f}\n")
        
        print(f"Summary results saved to: {csv_file}")
        
        # 保存完整结果到pickle文件
        pkl_file = os.path.join(output_dir, f"knn_full_results_{self.dataset}_{timestamp}.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Full results saved to: {pkl_file}")
    
    def print_results(self, results: Dict):
        """
        打印测试结果
        
        Args:
            results: 测试结果
        """
        print("\n" + "=" * 70)
        print("kNN TEST RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\nDataset: {results['dataset']}")
        print(f"Training samples: {results['train_samples']}")
        print(f"Validation samples: {results['val_samples']}")
        print(f"Number of classes: {results['num_classes']}")
        print(f"Feature dimension: {results['feature_dim']}")
        print(f"Total test time: {results['test_time']:.2f} seconds\n")
        
        print("-" * 70)
        print(f"{'k':<6} {'Classification Acc':<20} {'Retrieval Acc':<15}")
        print("-" * 70)
        
        for k, k_results in sorted(results['results'].items()):
            print(f"{k:<6} {k_results['classification_accuracy']:<20.4f} "
                  f"{k_results['retrieval_accuracy']:<15.4f}")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Parallel kNN Classification and Retrieval Test')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., Cifar100-LT)')
    parser.add_argument('--max_k', type=int, default=10,
                        help='Maximum k value to test')
    parser.add_argument('--results_dir', type=str, default='/home/Users/dqy/Projects/ConCutMix/results',
                        help='Base results directory')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for processing')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached features if available')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU acceleration (requires FAISS-GPU)')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false',
                        help='Do not use cache')
    parser.set_defaults(use_cache=True)
    
    args = parser.parse_args()
    
    try:
        # 创建性能测试器
        tester = KNNPerformanceTester(
            results_dir=args.results_dir,
            dataset=args.dataset,
            use_gpu=args.use_gpu
        )
        
        # 运行测试
        results = tester.run_test(
            max_k=args.max_k,
            max_workers=args.max_workers,
            use_cache=args.use_cache,
            batch_size=args.batch_size
        )
        
        # 打印结果
        tester.print_results(results)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    # 设置多进程启动方法（对于某些系统可能需要）
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    main()