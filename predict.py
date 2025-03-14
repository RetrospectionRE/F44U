import os
import torch
import time
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from model import Unet
from res18 import ResNet18_UNet
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor
from Punet import PUnet
from nets.rcnet import RCNet
from nets.renet import RENet
from nets.rknet import RKNet
from nets.rpnet import RPNet
from nets.segnet import SegNet

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_single_image(net, image_path, device, transform, save_dir=None):
    net.eval()
    try:
        with Image.open(image_path).convert('RGB') as image:
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    out_img = net(img_tensor)

                pred_prob = torch.sigmoid(out_img).squeeze().cpu().numpy()

                # 确保预测是二维数组
                if len(pred_prob.shape) > 2:
                    pred_prob = pred_prob[0]

                if pred_prob.size == 0:
                    raise ValueError("Prediction array is empty.")

                if pred_prob.ndim != 2:
                    raise ValueError(f"Unexpected number of dimensions in pred_prob: {pred_prob.ndim}")

                pred_prob = pred_prob.astype(np.float32)
                pred_prob_resized = cv2.resize(pred_prob, (image.width, image.height), interpolation=cv2.INTER_LINEAR)

                threshold = 0.5
                pred = (pred_prob_resized > threshold).astype(np.uint8)

                colored_pred = apply_color_map(pred, [(0, 0, 0), (255, 0, 0)])

                # 将预测结果直接绘制到原始图像上
                original_image_np = np.array(image)
                output_image = original_image_np.copy()
                mask = np.stack((pred,) * 3, axis=-1)  # Convert single channel mask to 3 channels
                mask = mask.astype(bool)
                output_image[mask] = colored_pred[mask]  # Apply the color only where there's a prediction

                output_image = Image.fromarray(output_image)

                pred_area_ratio = np.mean(pred)  # 计算预测为正类别的像素比例

                if pred_area_ratio > 1:
                    logging.info(
                        f'Skipping {image_path} due to prediction area ratio {pred_area_ratio:.2f} being greater than 0.8.')
                else:
                    if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, Path(image_path).stem + '.jpg')
                        output_image.save(save_path, "JPEG")
                        logging.info(f'Saved {save_path}')

            return output_image

    except Exception as e:
        logging.error(f'Error processing {image_path}: {e}')
        raise


def apply_color_map(prediction, color_map):
    """Apply a color map to the binary prediction."""
    h, w = prediction.shape[:2]
    colored_prediction = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(color_map):
        colored_prediction[prediction == class_id] = color
    return colored_prediction


if __name__ == "__main__":
    model_type = 'RKNet'  # Choose between 'Unet', 'CBAMUnet', and 'ResNet18_UNet'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the selected model
    if model_type == 'Unet':
        net = Unet(num_class=1)

    elif model_type == 'ResNet18_UNet':
        net = ResNet18_UNet(n_classes=1)

    elif model_type == 'PUnet':
        net = PUnet(1)
    elif model_type == 'RCNet':
        net = RCNet(1)
    elif model_type == 'RENet':
        net = RENet(1)
    elif model_type == 'RKNet':
        net = RKNet(1)
    elif model_type == 'RPNet':
        net = RPNet(1)
    elif model_type == 'SegNet':
        net = SegNet()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    net.to(device)

    # Load the model weights
    model_path = r"D:\LUNWEN\data2025\RKN\best_model_epoch.pth"
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)

    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4765, 0.4967, 0.4653], std=[0.2043, 0.1905, 0.1740])

    ])

    mode = "dir_predict"
    dir_origin_path = r"D:\jinzhan\outputs\restore1"
    dir_save_path = r"D:\jinzhan\outputs\restore1"

    if mode == "dir_predict":
        os.makedirs(dir_save_path, exist_ok=True)


        def process_file(filename):
            img_path = os.path.join(dir_origin_path, filename)
            predict_single_image(net, img_path, device, image_transform, save_dir=dir_save_path)


        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust number of workers as needed
            futures = [executor.submit(process_file, filename) for filename in os.listdir(dir_origin_path) if
                       filename.lower().endswith('.jpg')]
            for future in futures:
                future.result()  # This will raise any exceptions that occurred during execution

    elif mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
                r_image = predict_single_image(net, img, device, image_transform)
                r_image.show()
            except Exception as e:
                logging.error(f'Open Error! Try again!: {e}')