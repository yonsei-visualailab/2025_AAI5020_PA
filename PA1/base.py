import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from model import NetVLADModel
from train import train, validate
from eval import evaluate_recall

class LoadDataset(Dataset):
    def __init__(self, query_file, db_file, root_dir, gt_npz, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        with open(query_file, 'r') as f:
            self.query_images = [os.path.join(root_dir, line.strip()) for line in f]
        with open(db_file, 'r') as f:
            self.db_images = [os.path.join(root_dir, line.strip()) for line in f]

        gt = np.load(gt_npz, allow_pickle=True)
        self.utmQ = gt["utmQ"]
        self.utmDb = gt["utmDb"]
        self.posDistThr = float(gt["posDistThr"])

    def __len__(self): return len(self.query_images)

    def __getitem__(self, idx):
        q_path = self.query_images[idx]
        q_img = Image.open(q_path).convert("RGB")

        dists = np.linalg.norm(self.utmDb - self.utmQ[idx], axis=1)

        pos_indices = np.where(dists < self.posDistThr)[0]
        if len(pos_indices) == 0: pos_indices = [np.argmin(dists)]
        p_path = self.db_images[np.random.choice(pos_indices)]
        p_img = Image.open(p_path).convert("RGB")

        neg_indices = np.where(dists > 10 * self.posDistThr)[0]
        if len(neg_indices) == 0: neg_indices = [np.argmax(dists)]
        n_path = self.db_images[np.random.choice(neg_indices)]
        n_img = Image.open(n_path).convert("RGB")

        if self.transform:
            q_img = self.transform(q_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        return q_img, p_img, n_img

class ImageListDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None):
        with open(list_file, "r") as f:
            self.paths = [os.path.join(root_dir, line.strip()) for line in f]
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, idx, p

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = LoadDataset(
        query_file="dataset/query_train.txt",
        db_file="dataset/index_train.txt",
        root_dir="",
        gt_npz="dataset/gt/pitts30k_train.npz",
        transform=transform
    )

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    # -------------------------------
    # TASK1 : VGG16-based NetVLAD Structure
    # -------------------------------
    model = NetVLADModel(num_clusters=16).to(device)
    

    # -------------------------------
    # TASK2 : Learning with Ranking Loss
    # -------------------------------
    train(model, train_loader, val_loader, device, epochs=5)
    test_loss = validate(model, test_loader, device)
    print(f"Final Test Loss = {test_loss:.4f}")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/netvlad_final.pth")
    

    # -------------------------------
    # TASK3 : Evaluation with Recall@K
    # -------------------------------
    gt = np.load("dataset/gt/pitts30k_val.npz", allow_pickle=True)
    query_ds = ImageListDataset("dataset/query_val.txt", "", transform)
    db_ds = ImageListDataset("dataset/index_val.txt", "", transform)
    evaluate_recall(model, query_ds, db_ds, gt["utmQ"], gt["utmDb"], float(gt["posDistThr"]), device, recall_values=[1,5,10])


if __name__ == "__main__":
    main()
