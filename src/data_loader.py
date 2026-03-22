import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, transform=None, is_train=True):
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.is_train = is_train
        
        # Build file list
        if is_train:
            self.image_paths = sorted(glob.glob(os.path.join(root_dir, category, "train", "good", "*.png")))
            self.labels = [0] * len(self.image_paths) # 0 = Normal
        else:
            # Test folder contains 'good' and various anomaly subfolders
            test_path = os.path.join(root_dir, category, "test")
            self.image_paths = []
            self.labels = []
            for subfolder in sorted(os.listdir(test_path)):
                sub_path = os.path.join(test_path, subfolder)
                if os.path.isdir(sub_path):
                    paths = glob.glob(os.path.join(sub_path, "*.png"))
                    self.image_paths.extend(paths)
                    # Label: 0 for good, 1 for everything else
                    self.labels.extend([0 if subfolder == "good" else 1] * len(paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class MVTecDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, category, batch_size=32, img_size=256):
        super().__init__()
        self.root_dir = root_dir
        self.category = category
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # No ImageNet normalization here; we want raw [0,1] for Reconstruction
        ])

    def setup(self, stage=None):
        self.train_ds = MVTecDataset(self.root_dir, self.category, self.transform, is_train=True)
        self.test_ds = MVTecDataset(self.root_dir, self.category, self.transform, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        # We use the test set for validation to track Anomaly Detection metrics during training
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2)