import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Unet
from CBAM import CBAMUnet
from res18 import ResNet18_UNet
from dataset import CustomDataset
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from aspp import AUnet
from Punet import PUnet
from nets.rcnet import RCNet
from nets.renet import RENet
from nets.rknet import RKNet
from nets.rpnet import RPNet
from nets.segnet import SegNet
from nets.rc34 import R34CNet
from nets.rk34 import R34KNet
from nets.rc50 import R50CNet
from nets.rk50 import R50KNet
from nets.rc101 import R101CNet
from nets.rk101 import R101KNet
# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据预处理
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4765, 0.4967, 0.4653], std=[0.2043, 0.1905, 0.1740])
])
label_transform = transforms.Compose([
    transforms.Resize((256, 256)),
     # 注意这里是一个元组 (height, width)
    transforms.ToTensor()
]) # 转换标签为张量
# 读取验证集文件名
def read_filenames(file_path):
    with open(file_path, 'r') as f:
        filenames = [line.strip() for line in f]
    return filenames


class Evaluator:
    def __init__(self, model_type, model_path, device='cuda'):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        if self.model_type == 'Unet':
            net = Unet(1).to(self.device)
        elif self.model_type == 'CBAMUnet':
            net = CBAMUnet(1).to(self.device)
        elif self.model_type == 'ResNet18_UNet':
            net = ResNet18_UNet(1).to(self.device)
        elif model_type == 'AUnet':
            net = AUnet(1).to(self.device)
        elif model_type == 'PUnet':
            net = PUnet(1).to(device)
        elif model_type == 'RCNet':
            net = RCNet(1).to(self.device)
        elif model_type =='RENet':
            net = RENet(1).to(self.device)
        elif model_type == 'RKNet':
            net = RKNet(1).to(self.device)
        elif model_type == 'RPNet':
            net = RPNet(1).to(self.device)
        elif model_type == 'SegNet':
            net = SegNet().to(self.device)
        elif model_type == 'RC34':
            net = R34CNet(1).to(self.device)
        elif model_type == 'RC50':
            net = R50CNet(1).to(self.device)
        elif model_type == 'RC101':
            net = R101CNet(1).to(self.device)
        elif model_type == 'RK34':
            net = R34KNet(1).to(self.device)
        elif model_type == 'RK50':
            net = R50KNet(1).to(self.device)
        elif model_type == 'RVK101':
            net = R101KNet(1).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        return net

    def evaluate(self, dataloader):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader, 1):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                outputs = self.model(images)
                preds = torch.sigmoid(outputs) > 0.5  # 转换为二进制输出
                preds = F.interpolate(preds.float(), size=(256, 256), mode='nearest')
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                if batch_idx % 10 == 0:
                    logging.info(f"Processed batch {batch_idx}/{len(dataloader)}")

        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        # 确保所有预测值和标签都是布尔型或者整数类型的二进制值
        all_preds = all_preds.astype(np.uint8)
        all_labels = all_labels.astype(np.uint8)

        metrics = self.calculate_metrics(all_labels, all_preds)
        self.plot_roc_curve(all_labels, all_preds)
        self.plot_confusion_matrix(all_labels, all_preds)
        self.save_metrics(metrics)
        return metrics
    def calculate_metrics(self, labels, preds):
        precision = precision_score(labels, preds, average='binary', zero_division=0)
        recall = recall_score(labels, preds, average='binary', zero_division=0)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def plot_roc_curve(self, labels, preds):
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join('./logs', 'roc_curve.png'))
        plt.close()

    def plot_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join('./logs', 'confusion_matrix.png'))
        plt.close()

    def save_metrics(self, metrics):
        metrics_path = os.path.join('./logs', 'metrics.txt')
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")

if __name__ == '__main__':
    # 加载数据集
    dataset_path = r'.\DATA'
    val_filenames = r'.\DATA\class\test_files.txt'
    val_filenames = read_filenames(val_filenames)

    # 创建验证集
    dataset = CustomDataset(
        path=dataset_path,
        image_folder='image',
        label_folder='mask',
        transform=image_transform,
        label_transform=label_transform
    )

    val_indices = [i for i, (img_name, _) in enumerate(dataset.pairs) if img_name in val_filenames]
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model_type = 'RC101'  # 选择模型类型
    model_path = r'.\logs\best_model_epoch.pth'

    # 初始化评估器
    evaluator = Evaluator(model_type=model_type, model_path=model_path)

    # 评估模型
    logging.info("Starting evaluation...")
    metrics = evaluator.evaluate(val_loader)
    logging.info("Evaluation completed.")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")