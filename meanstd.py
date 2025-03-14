from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
class CustomDataset(Dataset):
    def __init__(self, path, image_folder='image', label_folder='mask', transform=None, label_transform=None):
        self.path = path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform if label_transform is not None else transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        # Get image file list and ensure sorting
        image_files = sorted(os.listdir(os.path.join(path, self.image_folder)))
        label_files = sorted(os.listdir(os.path.join(path, self.label_folder)))

        # Check if filenames match and create pairs list
        self.pairs = []
        for img_name in image_files:
            label_name = img_name.replace('.jpg', '.png')
            if label_name in label_files:
                self.pairs.append((img_name, label_name))
            else:
                print(f"Warning: No matching label found for image {img_name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_name, label_name = self.pairs[index]

        image_path = os.path.join(self.path, self.image_folder, img_name)
        label_path = os.path.join(self.path, self.label_folder, label_name)

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # Ensure label is single channel

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
        if self.label_transform is not None:
            label = self.label_transform(label)

        # Debugging information (optional)
        print(f"Image type: {type(image)}, Label type: {type(label)}")

        return image, label

# Define the transformation for images: resize, convert to tensor, and normalize
data_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Load the dataset and apply the transformation
dataset = CustomDataset(
    path=r'D:\U-NET_Origin\DATA',
    image_folder='image',
    label_folder='mask',
    transform=data_transform,
)

# Create a data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)

# Initialize variables to store the sum and sum of squares
sum_ = torch.zeros(3, dtype=torch.float64)  # Use double precision for numerical stability
sum_sq = torch.zeros(3, dtype=torch.float64)
nb_pixels = 0

for image, _ in dataloader:  # Ignore the label part
    batch_samples = image.size(0)
    C, H, W = image.size()[1:]  # Get the number of channels, height, and width

    sum_ += image.sum(dim=[0, 2, 3])  # Sum along the B and H*W dimensions
    sum_sq += (image ** 2).sum(dim=[0, 2, 3])  # Sum of squares along the B and H*W dimensions
    nb_pixels += batch_samples * H * W

# Calculate mean and standard deviation
mean = sum_ / nb_pixels
std = torch.sqrt(sum_sq / nb_pixels - mean ** 2 + 1e-8)  # Prevent division by zero

print(f"Mean: {mean}, Std: {std}")