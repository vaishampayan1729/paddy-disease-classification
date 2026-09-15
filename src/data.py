#====================================================
# import libraries
#====================================================
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#====================================================
# define the project path
#====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = DATA_DIR / "train_images"
TEST_DIR = DATA_DIR / "test_images"

#====================================================
# create training and validation split
#====================================================
train_dataset_full = datasets.ImageFolder(TRAIN_DIR)

# class information
NUM_CLASSES = len(train_dataset_full.classes)
CLASS_NAMES = train_dataset_full.classes

# create the stratified split for training and validation datasets
train_size_ratio = 0.8
generator = torch.Generator().manual_seed(42)

targets = torch.tensor(train_dataset_full.targets)

train_indices = []
val_indices = []

for class_idx in range(len(train_dataset_full.classes)):
    class_indices = torch.where(targets == class_idx)[0]

    permutation = torch.randperm(
        len(class_indices),
        generator=generator
    )

    class_indices = class_indices[permutation]

    train_count = round(train_size_ratio * len(class_indices))

    train_indices.extend(class_indices[:train_count].tolist())
    val_indices.extend(class_indices[train_count:].tolist())

train_dataset = torch.utils.data.Subset(
    train_dataset_full,
    train_indices
)

val_dataset = torch.utils.data.Subset(
    train_dataset_full,
    val_indices
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

#====================================================
# define the transformations for the training and validation datasets
#====================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# define the transformations for the training dataset
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# define the transformations for the validation dataset
val_transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

#====================================================
# create the training and validation datasets with 
# the defined transformations
#====================================================
train_dataset_full_transformed = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset_full_transformed = datasets.ImageFolder(
    TRAIN_DIR,
    transform=val_transform
)

train_dataset = torch.utils.data.Subset(
    train_dataset_full_transformed,
    train_dataset.indices
)

val_dataset = torch.utils.data.Subset(
    val_dataset_full_transformed,
    val_dataset.indices
)

#====================================================
# Create Data Loaders for train and val sets
#====================================================
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

