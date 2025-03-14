import os
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, path, image_folder='image', label_folder='mask', transform=None, label_transform=None):
        self.path = path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform

        # 获取图像和标签文件名并确保匹配
        image_files = sorted(os.listdir(os.path.join(path, self.image_folder)))
        label_files = sorted(os.listdir(os.path.join(path, self.label_folder)))

        self.pairs = []
        for img_name in image_files:
            label_name = img_name.replace('.jpg', '.png')
            if label_name in label_files:
                self.pairs.append((img_name, label_name))
            else:
                print(f"警告: 未找到与图像 {img_name} 匹配的标签文件")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_name, label_name = self.pairs[index]

        image_path = os.path.join(self.path, self.image_folder, img_name)
        label_path = os.path.join(self.path, self.label_folder, label_name)

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 确保标签是单通道

        # Resize images and labels to a consistent size
        image = image.resize((512, 512))
        label = label.resize((512, 512))

        # 应用数据增强（如果定义了）
        if self.transform is not None:
            seed = np.random.randint(2025123456)  # 生成随机种子
            torch.manual_seed(seed)  # 对图像变换应用种子
            image = self.transform(image)
            torch.manual_seed(seed)  # 对标签变换应用相同的种子
            label = self.label_transform(label)

        return image, label

# 新增的分布打印函数 ------------------------------------------------
def print_distribution(subset, class_labels, name):
    """打印子集的类别分布"""
    subset_classes = [class_labels[i] for i in subset.indices]
    unique, counts = np.unique(subset_classes, return_counts=True)
    print(f"\n{name} 分布（样本数: {len(subset)}）:")
    for cls, cnt in zip(unique, counts):
        print(f"  类别 {cls}: {cnt} 样本 ({cnt/len(subset):.2%})")

if __name__ == '__main__':
    dataset_path = r'D:\U-NET_Origin\DATA'
    full_dataset = CustomDataset(dataset_path)

    # 生成主类别标签列表
    class_labels = []
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        if isinstance(label, Image.Image):
            label = np.array(label)
        elif isinstance(label, torch.Tensor):
            label = label.numpy()
        unique, counts = np.unique(label, return_counts=True)
        main_class = unique[np.argmax(counts)]
        class_labels.append(main_class)

    # 分层分割数据集
    train_idx, temp_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=0.4,
        stratify=class_labels,
        random_state=7039664
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=[class_labels[i] for i in temp_idx],
        random_state=7039664
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # 保存文件名到txt文件
    def save_filenames_to_file(subset, filename):
        files = [full_dataset.pairs[i][0] for i in subset.indices]
        with open(filename, 'w') as f:
            for file in files:
                f.write(f"{file}\n")

    save_filenames_to_file(train_dataset, r'D:\U-NET_Origin\DATA\class\train_files.txt')
    save_filenames_to_file(val_dataset, r'D:\U-NET_Origin\DATA\class\val_files.txt')
    save_filenames_to_file(test_dataset, r'D:\U-NET_Origin\DATA\class\test_files.txt')

    # 打印分布时传入class_labels参数
    print_distribution(train_dataset, class_labels, "训练集")
    print_distribution(val_dataset, class_labels, "验证集")
    print_distribution(test_dataset, class_labels, "测试集")