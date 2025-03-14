import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from CBAM import CBAMUnet
from dataset import CustomDataset
from model import Unet
from res18 import ResNet18_UNet
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

from torch.optim.lr_scheduler import ReduceLROnPlateau  # 修改为自适应学习率调度器
from torch.cuda.amp import GradScaler, autocast

# 设置随机种子以保证结果的可复现性
torch.manual_seed(10965590)

# 定义超参数
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-2
num_epochs = 300
save_interval = 10
save_path = r'D:\U-NET_Origin\logs'
os.makedirs(save_path, exist_ok=True)
model_type = 'RKNet'  # 可选 'Unet', 'CBAMUnet', 'ResNet18_UNet'
optimizer_name = 'AdamW'  # 可选 'SGD', 'Adam', 'AdamW'
patience = 15  # Early stopping patience 15默认
min_lr = 1e-7  # 最小学习率
def read_filenames(file_path):
    """读取文件名列表"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

# 数据增强配置
train_image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4701, 0.4906, 0.4549],std=[0.2277, 0.2189, 0.2008])
    transforms.Normalize(mean=[0.4847, 0.4434, 0.4022],std=[0.2739, 0.2682, 0.2753])
])

train_label_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

val_image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4765, 0.4967, 0.4653], std=[0.2043, 0.1905, 0.1740])
    transforms.Normalize(mean=[0.4847, 0.4434, 0.4022],std=[0.2739, 0.2682, 0.2753])

])

val_label_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 初始化数据集和数据加载器
dataset = CustomDataset(
    path=r"D:\U-NET_Origin\DATA",
    image_folder='image',
    label_folder='mask',
    transform=None,
    label_transform=None
)

# 创建训练集和验证集
train_filenames = read_filenames(r'D:\U-NET_Origin\DATA\class\train_files.txt')
val_filenames = read_filenames(r'D:\U-NET_Origin\DATA\class\val_files.txt')

train_indices = [i for i, (img, _) in enumerate(dataset.pairs) if img in train_filenames]
val_indices = [i for i, (img, _) in enumerate(dataset.pairs) if img in val_filenames]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# 应用数据增强
train_dataset.dataset.transform = train_image_transform
train_dataset.dataset.label_transform = train_label_transform
val_dataset.dataset.transform = val_image_transform
val_dataset.dataset.label_transform = val_label_transform

# 创建数据加载器
def custom_collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

# 模型初始化
model_dict = {
    'Unet': Unet(1, dropout_prob=0),
    'CBAMUnet': CBAMUnet(1),
    'ResNet18_UNet': ResNet18_UNet(1, dropout_prob=0.2),
    'AUnet': AUnet(1),
    'PUnet': PUnet(1),
    'RCNet': RCNet(1,dropout_prob=0.5),
    'RENet': RENet(1,dropout_prob=0.5),
    'RKNet': RKNet(1),
    'RPNet': RPNet(1),
    'SegNet': SegNet(),
    'RC34': R34CNet(1),
    'RC50': R50CNet(1),
    'RC101': R101CNet(1),
    'RK34': R34KNet(1),
    'RK50': R50KNet(1),
    'RK101': R101KNet(1)
}

# 模型权重初始化
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

net = model_dict[model_type].to(device)
net.apply(init_weights)

# 加载预训练权重（可选）
pretrained_weights_path = r""
if os.path.exists(pretrained_weights_path):
    print("加载预训练权重...")
    net.load_state_dict(torch.load(pretrained_weights_path, map_location=device))

# 优化器配置
optimizer_dict = {
    'SGD': optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9),
    'Adam': optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4),
    'AdamW': optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-5)
}
optimizer = optimizer_dict[optimizer_name]

# 使用自适应学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=min_lr)

# 损失函数和混合精度训练
ratio = 1
pos_weight_val = ratio / 1
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))
scaler = torch.amp.GradScaler()

# 早停机制参数
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = ""

# 训练监控数据
train_losses, val_losses, lr_history = [], [], []

# IoU计算函数
def calculate_iou(preds, targets):
    intersection = (preds & targets).float().sum()
    union = (preds | targets).float().sum()
    return (intersection + 1e-6) / (union + 1e-6)

if __name__ == '__main__':
    start_time = time.time()

    try:
        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            net.train()
            epoch_train_loss = 0.0
            with tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as pbar:
                for batch_idx, (images, masks) in enumerate(pbar):
                    images, masks = images.to(device), masks.to(device).float()

                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        outputs = net(images)
                        outputs = F.interpolate(outputs, size=masks.shape[2:],
                                                mode='bilinear', align_corners=True)
                        loss = loss_fn(outputs, masks)

                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_train_loss += loss.item() * images.size(0)
                    pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = epoch_train_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            # 验证阶段
            net.eval()
            epoch_val_loss = 0.0
            correct_pixels = 0
            total_pixels = 0
            total_iou = 0.0
            with torch.no_grad(), tqdm(val_loader, desc='Validating') as pbar:
                for images, masks in pbar:
                    images, masks = images.to(device), masks.to(device).float()
                    with torch.amp.autocast('cuda'):
                        outputs = net(images)
                        outputs = F.interpolate(outputs, size=masks.shape[2:],
                                                mode='bilinear', align_corners=True)
                        loss = loss_fn(outputs, masks)

                    epoch_val_loss += loss.item() * images.size(0)

                    preds = torch.sigmoid(outputs) > 0.5
                    correct_pixels += (preds == masks.byte()).sum().item()
                    total_pixels += masks.numel()
                    total_iou += calculate_iou(preds, masks.byte()).item()
                    pbar.set_postfix({'val_loss': loss.item()})

            avg_val_loss = epoch_val_loss / len(val_loader.dataset)
            accuracy = correct_pixels / total_pixels
            avg_iou = total_iou / len(val_loader)
            val_losses.append(avg_val_loss)

            # 更新学习率（自适应）
            scheduler.step(avg_val_loss)

            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

            # 早停机制处理
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_path = os.path.join(save_path, f"best_model_epoch.pth")
                torch.save(net.state_dict(), best_model_path)
                print(f"✅ 发现新的最佳模型 (val_loss={avg_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"⚠️ 连续 {epochs_no_improve} 个epoch未改善")

            if epochs_no_improve >= patience:
                print(f"⛔️ 早停触发：连续 {patience} 个epoch未改善")
                net.load_state_dict(torch.load(best_model_path))
                break

            # 定期保存模型
            if epoch % save_interval == 0:
                checkpoint_path = os.path.join(save_path, f"model_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_train_loss,
                }, checkpoint_path)

            # 打印训练信息
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")
            print(f"  像素准确率: {accuracy:.4f} | IoU: {avg_iou:.4f} | 当前学习率: {current_lr:.2e}")

            # 可视化训练曲线
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(lr_history, label='Learning Rate', color='green')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'training_metrics.png'), dpi=300)
            plt.close()

            # 保存数据日志
            filename = f'training_log_{model_type}.txt'
            with open(os.path.join(save_path, filename), 'a') as f:
                f.write(f"{epoch},{avg_train_loss:.6f},{avg_val_loss:.6f},{accuracy:.6f},{avg_iou:.6f},{current_lr:.6e}\n")

    finally:
        total_time = time.time() - start_time
        print(f"\n🏁 训练完成！总用时: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.2f}s")
        print(f"最佳模型保存于: {best_model_path}")