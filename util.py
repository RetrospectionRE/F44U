from PIL import Image
import numpy as np
import torch
def keep_datasize_same(path,size=(512,512)):
    img=Image.open(path)
    length=max(img.size)
    mask=Image.new('RGB',(length,length),(0,0,0))
    mask.paste(img,(0,0))
    mask=mask.resize(size=size)
    return mask



def calculate_iou(output, target, n_classes):
    ious = []
    _, pred=torch.max(output,1)
    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()  # True Positive
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # 若没有预测或真实的该类，设置为nan
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return ious


def mean_iou(pred, target, n_classes):
    ious = calculate_iou(pred, target, n_classes)
    valid_ious = [iou for iou in ious if iou >= 0]
    mean_iou = sum(valid_ious) / len(valid_ious)
    return mean_iou


def calculate_accuracy(y_pred, y_true):
    """计算准确率"""
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    accuracy = correct / (y_true.size(1)*y_true.size(2))
    return accuracy


